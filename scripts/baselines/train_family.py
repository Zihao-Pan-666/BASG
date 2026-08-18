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
import re
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from basg.data.dataset import collate_prefix
from basg.features.catalog_semantic import load_catalog_embedding_pool
from basg.features.semantic import load_semantic_embeddings
from basg.models.bert4rec_family import BERT4RecSemanticFamily
from basg.training.bert4rec_trainer import train_bert4rec_family
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
    format_seconds,
    log_event,
    maybe_print_json,
    resolve_runtime_config,
    save_training_timing_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train protocol-matched BERT4Rec Sem/RecG/SAGE baselines."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override data.seed and update seed-dependent paths")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-progress", "--no_progress", action="store_true", dest="no_progress")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--progress-refresh", type=float, default=None)
    return parser.parse_args()


def _apply_seed_override(cfg: dict, new_seed: int) -> None:
    """Override data.seed and replace seed in all path values."""
    old = str(cfg.get("data", {}).get("seed", ""))
    cfg["data"]["seed"] = int(new_seed)
    if old:
        for key, value in cfg.get("paths", {}).items():
            if isinstance(value, str):
                cfg["paths"][key] = re.sub(rf"seed{old}", f"seed{new_seed}", value)
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str):
            cfg["paths"][key] = value.replace("{seed}", str(new_seed))


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header and header != fieldnames:
            raise RuntimeError(
                f"CSV schema mismatch for {path}. Existing={header}, new={fieldnames}"
            )

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg = copy.deepcopy(cfg)
    if args.seed is not None:
        _apply_seed_override(cfg, int(args.seed))
    if args.alpha is not None:
        cfg.setdefault("alignment", {})["alpha"] = float(args.alpha)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.max_train_batches is not None:
        cfg.setdefault("training", {})["max_batches_per_epoch"] = int(
            args.max_train_batches
        )
    if args.max_val_batches is not None:
        cfg.setdefault("evaluation", {})["max_batches"] = int(args.max_val_batches)
    if args.run_name is not None:
        cfg.setdefault("experiment", {})["run_name"] = str(args.run_name)

    runtime = resolve_runtime_config(
        cfg.get("runtime"),
        no_progress=args.no_progress,
        print_json=args.print_json,
        progress_refresh=args.progress_refresh,
    )
    cfg["runtime"] = runtime
    clock = RunClock()

    experiment = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = dict(cfg["training"])
    alignment_cfg = cfg.get("alignment", {})
    evaluation_cfg = dict(cfg["evaluation"])

    mode = str(experiment["mode"]).lower()
    architecture = str(experiment["architecture"]).lower()
    run_name = str(experiment["run_name"])
    source = str(cfg["source"])
    aux_domains = list(cfg.get("aux_domains", []))
    data_root = str(cfg.get("data_root", "./data"))
    embedding_dir = str(data_cfg.get("embedding_dir", "semantic_embeddings"))
    seed = int(data_cfg.get("seed", 2026))

    expected_architecture = "single" if mode == "sem" else "dual"
    if architecture != expected_architecture:
        raise ValueError(
            f"mode={mode} requires architecture={expected_architecture}; "
            f"got {architecture}"
        )

    set_seed_compat(seed)
    device = resolve_device_compat(str(training_cfg.get("device", "auto")))
    log_event(
        "RUN",
        "START",
        stage="baseline_train",
        run=run_name,
        mode=mode,
        architecture=architecture,
        source=source,
        seed=seed,
        device=device,
    )

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
        data_root=data_root,
        domain=source,
        item_map=item_map,
        embedding_dir=embedding_dir,
    )
    if int(item_features.shape[0]) != len(item_map) + 1:
        raise RuntimeError(
            "Source feature table and item map disagree: "
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
    log_event(
        "DATA",
        "READY",
        source=source,
        users=len(user_map),
        items=len(item_map),
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )

    auxiliary_pools: dict[str, torch.Tensor] = {}
    auxiliary_metadata: list[dict[str, object]] = []
    if mode in {"recg", "sage"}:
        for domain_index, domain in enumerate(aux_domains):
            pool, metadata = load_catalog_embedding_pool(
                data_root=data_root,
                embedding_dir=embedding_dir,
                domain=str(domain),
                pool_size=int(alignment_cfg.get("pool_size_per_domain", 4096)),
                seed=seed + 1000 + domain_index,
            )
            if int(pool.shape[1]) != int(item_features.shape[1]):
                raise ValueError(
                    f"Embedding dimension mismatch for auxiliary domain {domain}: "
                    f"{pool.shape[1]} vs source {item_features.shape[1]}"
                )
            auxiliary_pools[str(domain)] = pool
            auxiliary_metadata.append(metadata)
            log_event(
                "DATA",
                "AUX_CATALOG",
                domain=domain,
                pool_rows=int(pool.shape[0]),
                embedding_dim=int(pool.shape[1]),
            )

    model = BERT4RecSemanticFamily(
        item_features=item_features,
        hidden_dim=int(model_cfg["hidden_dim"]),
        max_len=int(data_cfg["max_len"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        architecture=architecture,
    ).to(device)

    training_cfg["train_negatives"] = int(data_cfg["train_negatives"])
    training_cfg["runtime"] = runtime
    evaluation_cfg["ranking"] = str(data_cfg["ranking"])
    evaluation_cfg["eval_negatives"] = int(data_cfg["eval_negatives"])

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    result_dir = Path(cfg["paths"]["result_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}_{source}_seed{seed}.pt"

    summary = train_bert4rec_family(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_items=len(item_map),
        mode=mode,
        auxiliary_pools=auxiliary_pools,
        checkpoint_path=checkpoint_path,
        training_cfg=training_cfg,
        alignment_cfg=alignment_cfg,
        evaluation_cfg=evaluation_cfg,
        seed=seed,
        source_domain=source,
        run_name=run_name,
    )
    chart_path = save_training_timing_chart(
        summary["history"],
        result_dir / f"{run_name}_{source}_seed{seed}_runtime.png",
        title=f"{run_name}: training time",
    )
    summary.update(
        {
            "config_path": str(Path(args.config)),
            "model_hparams": model.export_hparams(),
            "num_source_users": len(user_map),
            "num_source_items": len(item_map),
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "auxiliary_metadata": auxiliary_metadata,
            "runtime": {
                **runtime,
                "run_elapsed_seconds": float(clock.elapsed()),
                "timing_chart_path": chart_path,
            },
            "protocol": {
                "checkpoint_selection": "source_validation_only",
                "target_interactions_used_for_training": False,
                "target_interactions_used_for_model_selection": False,
                "left_padding_readout": "last_position",
                "candidate_projection": "recommendation_projection",
                "finite_fail_fast": True,
                "fixed_validation_candidates_across_epochs": True,
            },
        }
    )

    summary_path = result_dir / f"{run_name}_{source}_seed{seed}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    csv_row = {
        "run_name": run_name,
        "mode": mode,
        "architecture": architecture,
        "source": source,
        "seed": seed,
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "dropout": float(model_cfg["dropout"]),
        "best_epoch": summary["best_epoch"],
        "selection_metric": summary["best_metric_name"],
        "source_val_metric": summary["best_metric_value"],
        "checkpoint_path": summary["checkpoint_path"],
        "summary_path": str(summary_path),
    }
    append_csv(result_dir / "bert4rec_family_source_val.csv", csv_row)
    log_event("ARTIFACT", "SAVED", checkpoint=checkpoint_path, summary=summary_path)
    if chart_path:
        log_event("ARTIFACT", "TIMING_CHART", path=chart_path)
    log_event(
        "RUN",
        "DONE",
        stage="baseline_train",
        run=run_name,
        best_epoch=summary["best_epoch"],
        metric=f"{summary['best_metric_name']}:{summary['best_metric_value']:.6f}",
        elapsed=format_seconds(clock.elapsed()),
    )
    maybe_print_json(csv_row, enabled=bool(runtime["print_json"]))


if __name__ == "__main__":
    main()
