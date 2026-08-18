"""Train UniSRec baseline on source domain under strict zero-shot protocol.

Follows the same data pipeline and evaluation protocol as train_family.py.
UniSRec uses causal (SASRec-style) attention with MoE adaptor layers instead
of BERT4Rec's bidirectional attention, so it requires a standalone training
loop rather than reusing train_bert4rec_family().

Usage:
    python scripts/baselines/train_unisrec.py --config configs/baselines/unisrec.yaml --seed 2026
"""

from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from basg.baselines.unisrec import UniSRec
from basg.data.dataset import collate_prefix
from basg.evaluation.bert4rec_family_evaluator import evaluate_bert4rec_family
from basg.features.semantic import load_semantic_embeddings
from basg.utils.bert4rec_family_compat import (
    build_source_splits_compat,
    group_user_sequences_compat,
    load_interactions_compat,
    make_prefix_dataset,
    resolve_device_compat,
    set_seed_compat,
)
from basg.utils.runtime import (
    RunClock,
    close_progress,
    format_seconds,
    loader_total,
    log_event,
    maybe_print_json,
    progress_iter,
    resolve_progress_enabled,
    set_progress_postfix,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
        if header and header != fieldnames:
            raise RuntimeError(
                f"CSV schema mismatch for {path}. Existing={header}, new={fieldnames}"
            )
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _sample_train_negatives(
    histories: torch.Tensor,
    targets: torch.Tensor,
    num_items: int,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Source-only rejection sampler for BPR negatives."""
    history_rows = histories.detach().cpu().tolist()
    target_rows = targets.detach().cpu().tolist()
    output: list[list[int]] = []
    for history, target in zip(history_rows, target_rows):
        blocked = {int(x) for x in history if int(x) > 0}
        blocked.add(int(target))
        row: list[int] = []
        attempts = 0
        limit = max(100, count * 100)
        while len(row) < count and attempts < limit:
            item = int(rng.integers(1, num_items + 1))
            if item not in blocked:
                row.append(item)
            attempts += 1
        while len(row) < count:
            row.append(int(rng.integers(1, num_items + 1)))
        output.append(row)
    return torch.as_tensor(output, dtype=torch.long)


# ── Main ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train UniSRec baseline on source domain."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="Override data.seed.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-progress", action="store_true", dest="no_progress")
    p.add_argument("--print-json", action="store_true")
    p.add_argument("--progress-refresh", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    if args.seed is not None:
        cfg["data"]["seed"] = int(args.seed)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.max_train_batches is not None:
        cfg["training"]["max_batches_per_epoch"] = int(args.max_train_batches)
    if args.max_val_batches is not None:
        cfg["evaluation"]["max_batches"] = int(args.max_val_batches)
    if args.run_name is not None:
        cfg["experiment"]["run_name"] = str(args.run_name)

    # Runtime config
    rt_cfg = cfg.get("runtime", {})
    show_progress = resolve_progress_enabled(
        False if args.no_progress else rt_cfg.get("show_progress", True)
    )
    progress_refresh = float(
        args.progress_refresh
        if args.progress_refresh is not None
        else rt_cfg.get("progress_refresh_seconds", 0.5)
    )

    experiment = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = dict(cfg["training"])
    evaluation_cfg = dict(cfg["evaluation"])

    run_name = str(experiment["run_name"])
    source = str(cfg["source"])
    data_root = str(cfg.get("data_root", "./data"))
    embedding_dir = str(data_cfg.get("embedding_dir", "semantic_embeddings"))
    seed = int(data_cfg.get("seed", 2026))

    set_seed_compat(seed)
    device = resolve_device_compat(str(training_cfg.get("device", "auto")))

    log_event("RUN", "START", stage="unisrec_train", run=run_name,
              source=source, seed=seed, device=device)

    # ── Data ──────────────────────────────────────────────────────
    interactions, user_map, item_map = load_interactions_compat(data_root, source)
    sequences = group_user_sequences_compat(
        interactions, min_len=int(data_cfg["min_len"])
    )
    train_samples, val_samples = build_source_splits_compat(
        sequences=sequences,
        max_len=int(data_cfg["max_len"]),
        min_len=int(data_cfg["min_len"]),
    )
    item_features = load_semantic_embeddings(
        data_root=data_root, domain=source, item_map=item_map,
        embedding_dir=embedding_dir,
    )
    if int(item_features.shape[0]) != len(item_map) + 1:
        raise RuntimeError(
            f"Feature table / item map disagree: "
            f"{item_features.shape[0]} vs {len(item_map) + 1}"
        )

    train_dataset = make_prefix_dataset(
        train_samples, num_items=len(item_map), max_len=int(data_cfg["max_len"])
    )
    val_dataset = make_prefix_dataset(
        val_samples, num_items=len(item_map), max_len=int(data_cfg["max_len"])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_prefix,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed + 101),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_prefix,
        pin_memory=device.type == "cuda",
    )

    num_items = len(item_map)
    log_event("DATA", "READY", source=source, users=len(user_map),
              items=num_items, train_samples=len(train_samples),
              val_samples=len(val_samples))

    # ── Model ─────────────────────────────────────────────────────
    model = UniSRec(
        item_features=item_features,
        hidden_dim=int(model_cfg["hidden_dim"]),
        max_len=int(data_cfg["max_len"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        n_exps=int(model_cfg.get("n_exps", 8)),
    ).to(device)

    # ── Training setup ────────────────────────────────────────────
    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    train_negatives = int(data_cfg.get("train_negatives", 5))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    patience_limit = int(training_cfg.get("early_stop_patience", 10))
    max_batches = int(training_cfg.get("max_batches_per_epoch", 0))
    primary_metric = str(evaluation_cfg.get("primary_metric", "NDCG@10"))
    eval_seed = seed + int(evaluation_cfg.get("fixed_negative_seed_offset", 10000))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    result_dir = Path(cfg["paths"]["result_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}_{source}_seed{seed}.pt"

    best_metric = -float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict] = []
    clock = RunClock()

    log_event("TRAIN", "START", run=run_name, epochs=epochs,
              train_batches=loader_total(train_loader, max_batches),
              val_batches=loader_total(val_loader,
                                       int(evaluation_cfg.get("max_batches", 0))))

    # ── Training loop ─────────────────────────────────────────────
    log_event("TRAIN", "CACHE_WARMUP", run=run_name,
              step="projecting_item_table")
    model.eval()
    model._ensure_projected()
    log_event("TRAIN", "CACHE_READY", run=run_name)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        pairs = 0
        latest_grad_norm = 0.0
        rng = np.random.default_rng(seed + epoch * 104729)

        train_iter = progress_iter(
            train_loader,
            total=loader_total(train_loader, max_batches),
            run_name=run_name,
            phase="TRAIN",
            detail=f"epoch={epoch:03d}/{epochs:03d}",
            enabled=show_progress,
            refresh_seconds=progress_refresh,
        )
        try:
            for batch_idx, batch in enumerate(train_iter):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                hist_cpu = batch["history"]
                tgt_cpu = batch["target"]
                neg_cpu = _sample_train_negatives(
                    hist_cpu, tgt_cpu, num_items=num_items,
                    count=train_negatives, rng=rng,
                )

                histories = hist_cpu.to(device, non_blocking=True)
                targets = tgt_cpu.to(device, non_blocking=True).unsqueeze(1)
                negatives = neg_cpu.to(device, non_blocking=True)

                user_state = model.encode_sequence(histories)
                pos_scores = model.score_from_state(user_state, targets)
                neg_scores = model.score_from_state(user_state, negatives)

                loss = -torch.nn.functional.logsigmoid(
                    pos_scores - neg_scores
                ).mean()

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, batch={batch_idx}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip,
                    error_if_nonfinite=True,
                )
                latest_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()

                # Parameter sanity
                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        raise FloatingPointError(
                            f"Non-finite parameter: {name}"
                        )
                    p_abs = float(param.detach().abs().max().cpu())
                    if p_abs > 1000.0:
                        raise FloatingPointError(
                            f"Parameter exploded: {name} absmax={p_abs}"
                        )

                B = int(tgt_cpu.shape[0])
                loss_sum += float(loss.detach()) * B
                pairs += B
                set_progress_postfix(
                    train_iter,
                    loss=loss_sum / max(pairs, 1),
                    grad=latest_grad_norm,
                    examples=pairs,
                    elapsed=format_seconds(clock.elapsed()),
                )
        finally:
            close_progress(train_iter)

        if pairs == 0:
            raise RuntimeError("No training batches were processed.")

        train_seconds = float(time.perf_counter() - epoch_start)

        # ── Validation ────────────────────────────────────────────
        val_start = time.perf_counter()
        metrics = evaluate_bert4rec_family(
            model=model,
            loader=val_loader,
            device=device,
            num_items=num_items,
            ranking=str(data_cfg.get("ranking", "sampled")),
            eval_negatives=int(data_cfg.get("eval_negatives", 100)),
            ks=tuple(int(k) for k in evaluation_cfg.get("ks", [10, 20])),
            tie_policy=str(evaluation_cfg.get("tie_policy", "worst")),
            seed=eval_seed,
            max_batches=int(evaluation_cfg.get("max_batches", 0)),
            show_progress=show_progress,
            progress_desc=f"{run_name}:val",
            progress_refresh=progress_refresh,
            leave_progress=False,
        )
        val_seconds = float(time.perf_counter() - val_start)
        current = float(metrics[primary_metric])

        row = {
            "epoch": epoch,
            "loss": float(loss_sum / pairs),
            "grad_norm": float(latest_grad_norm),
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
            "epoch_seconds": float(time.perf_counter() - epoch_start),
            "elapsed_seconds": float(clock.elapsed()),
            **{f"val_{k}": float(v) for k, v in metrics.items()},
        }
        history.append(row)

        improved = current > best_metric + 1e-12
        if improved:
            best_metric = current
            best_epoch = epoch
            patience = 0
            _atomic_torch_save(
                {
                    "format_version": 2,
                    "run_name": run_name,
                    "source_domain": source,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_metric_name": primary_metric,
                    "best_metric_value": float(best_metric),
                    "model_hparams": model.export_hparams(),
                    "model_state": model.state_dict(),
                    "training_cfg": dict(training_cfg),
                    "evaluation_cfg": dict(evaluation_cfg),
                    "protocol": {
                        "training_domain": source,
                        "target_interactions_used_for_training": False,
                        "target_catalog_statistics_used": False,
                        "target_model_selection_used": False,
                    },
                    "history": list(history),
                },
                checkpoint_path,
            )
        else:
            patience += 1

        log_event(
            "TRAIN", "EPOCH",
            run=run_name,
            epoch=f"{epoch:03d}/{epochs:03d}",
            loss=row["loss"],
            validation_metric=f"{primary_metric}:{current:.6f}",
            best=f"{best_metric:.6f}@{best_epoch}",
            status="saved" if improved else f"patience_{patience}/{patience_limit}",
        )

        if patience >= patience_limit:
            log_event("TRAIN", "EARLY_STOP", run=run_name, epoch=epoch)
            break

    if best_epoch == 0:
        raise RuntimeError("No checkpoint was saved during training.")

    # ── Save summary ──────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "source_domain": source,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_metric_name": primary_metric,
        "best_metric_value": float(best_metric),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(clock.elapsed()),
        "history": history,
        "model_hparams": model.export_hparams(),
        "config_path": str(Path(args.config)),
        "num_source_users": len(user_map),
        "num_source_items": num_items,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "protocol": {
            "checkpoint_selection": "source_validation_only",
            "target_interactions_used_for_training": False,
            "target_interactions_used_for_model_selection": False,
            "candidate_scoring": "dot_product_moe_projection",
        },
    }

    summary_path = result_dir / f"{run_name}_{source}_seed{seed}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_row = {
        "run_name": run_name,
        "architecture": "unisrec",
        "source": source,
        "seed": seed,
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "num_layers": int(model_cfg["num_layers"]),
        "n_exps": int(model_cfg.get("n_exps", 8)),
        "dropout": float(model_cfg["dropout"]),
        "best_epoch": best_epoch,
        "selection_metric": primary_metric,
        "source_val_metric": float(best_metric),
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
    }
    _append_csv(result_dir / "unisrec_source_val.csv", csv_row)

    log_event("ARTIFACT", "SAVED", checkpoint=str(checkpoint_path),
              summary=str(summary_path))
    log_event("RUN", "DONE", stage="unisrec_train", run=run_name,
              best_epoch=best_epoch,
              metric=f"{primary_metric}:{best_metric:.6f}",
              elapsed=format_seconds(clock.elapsed()))
    maybe_print_json(csv_row, enabled=bool(args.print_json))


if __name__ == "__main__":
    main()
