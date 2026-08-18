"""Zero-shot evaluate a source-trained UniSRec checkpoint on target domains.

Uses the same evaluation protocol as eval_family.py (sampled-100, tie_policy=worst)
and the same output format (zero_shot.json).

Usage:
    python scripts/baselines/eval_unisrec.py --config configs/baselines/unisrec.yaml --seed 2026
"""

from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from basg.baselines.unisrec import UniSRec
from basg.data.dataset import collate_prefix
from basg.evaluation.bert4rec_family_evaluator import evaluate_bert4rec_family
from basg.features.semantic import load_semantic_embeddings
from basg.utils.bert4rec_family_compat import (
    build_target_samples_compat,
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
    resolve_progress_enabled,
    save_stage_timing_chart,
)


# ── Helpers ─────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot evaluate a source-trained UniSRec checkpoint."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="Override data.seed (must match training seed).")
    p.add_argument("--checkpoint", default=None,
                   help="Override default checkpoint path.")
    p.add_argument("--targets", default=None,
                   help="Comma-separated target domains (default: from config).")
    p.add_argument("--no-progress", action="store_true", dest="no_progress")
    p.add_argument("--print-json", action="store_true")
    p.add_argument("--progress-refresh", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rt_cfg = cfg.get("runtime", {})
    show_progress = resolve_progress_enabled(
        False if args.no_progress else rt_cfg.get("show_progress", True)
    )
    progress_refresh = float(
        args.progress_refresh
        if args.progress_refresh is not None
        else rt_cfg.get("progress_refresh_seconds", 0.5)
    )

    if args.seed is not None:
        cfg["data"]["seed"] = int(args.seed)

    experiment = cfg["experiment"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    evaluation_cfg = dict(cfg["evaluation"])
    source = str(cfg["source"])
    declared_targets = [str(x) for x in cfg["targets"]]
    targets = (
        [t.strip() for t in args.targets.split(",") if t.strip()]
        if args.targets
        else declared_targets
    )
    unknown = set(targets) - set(declared_targets)
    if unknown:
        raise ValueError(f"Targets not declared in config: {sorted(unknown)}")

    data_root = str(cfg.get("data_root", "./data"))
    embedding_dir = str(data_cfg.get("embedding_dir", "semantic_embeddings"))
    seed = int(data_cfg.get("seed", 2026))
    run_name = str(experiment["run_name"])

    default_ckpt = (
        Path(cfg["paths"]["checkpoint_dir"])
        / f"{run_name}_{source}_seed{seed}.pt"
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_ckpt
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))

    set_seed_compat(seed)
    device = resolve_device_compat(str(training_cfg.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint.get("source_domain") != source:
        raise RuntimeError(
            f"Checkpoint source={checkpoint.get('source_domain')} "
            f"but config source={source}"
        )

    result_dir = Path(cfg["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    clock = RunClock()

    log_event("RUN", "START", stage="unisrec_zero_shot",
              run=run_name, source=source, targets=targets,
              seed=seed, device=device)

    all_results: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []

    for target_idx, target in enumerate(targets):
        target_start = time.perf_counter()
        log_event("EVAL", "TARGET_START", run=run_name, target=target,
                  target_index=f"{target_idx + 1}/{len(targets)}")

        interactions, user_map, item_map = load_interactions_compat(
            data_root, target
        )
        sequences = group_user_sequences_compat(
            interactions, min_len=int(data_cfg["min_len"])
        )
        samples = build_target_samples_compat(
            sequences=sequences,
            max_len=int(data_cfg["max_len"]),
            min_len=int(data_cfg["min_len"]),
        )
        item_features = load_semantic_embeddings(
            data_root=data_root, domain=target, item_map=item_map,
            embedding_dir=embedding_dir,
        )

        hparams = checkpoint["model_hparams"]
        model = UniSRec(
            item_features=item_features,
            hidden_dim=int(hparams["hidden_dim"]),
            max_len=int(hparams["max_len"]),
            num_layers=int(hparams["num_layers"]),
            num_heads=int(hparams["num_heads"]),
            dropout=float(hparams["dropout"]),
            n_exps=int(hparams.get("n_exps", 8)),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        loader = DataLoader(
            make_prefix_dataset(
                samples, num_items=len(item_map),
                max_len=int(data_cfg["max_len"]),
            ),
            batch_size=int(training_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_prefix,
            pin_memory=device.type == "cuda",
        )

        metrics = evaluate_bert4rec_family(
            model=model,
            loader=loader,
            device=device,
            num_items=len(item_map),
            ranking=str(data_cfg["ranking"]),
            eval_negatives=int(data_cfg["eval_negatives"]),
            ks=tuple(int(k) for k in evaluation_cfg.get("ks", [10, 20])),
            tie_policy=str(evaluation_cfg.get("tie_policy", "worst")),
            seed=seed
            + int(evaluation_cfg.get("target_seed_offset", 20000))
            + target_idx,
            max_batches=int(evaluation_cfg.get("max_batches", 0)),
            show_progress=show_progress,
            progress_desc=f"{run_name}:{target}",
            progress_refresh=progress_refresh,
            leave_progress=bool(rt_cfg.get("leave_progress_bar", True)),
        )

        row: dict[str, object] = {
            "run_name": run_name,
            "architecture": "unisrec",
            "source": source,
            "target": target,
            "seed": seed,
            "source_best_epoch": checkpoint["best_epoch"],
            "source_selection_metric": checkpoint["best_metric_name"],
            "source_selection_value": checkpoint["best_metric_value"],
            "ranking": data_cfg["ranking"],
            "eval_negatives": int(data_cfg["eval_negatives"]),
            "target_interactions_used_for_training": False,
            "target_interactions_used_for_model_selection": False,
        }
        for key in (
            "Recall@10", "Recall@20", "NDCG@10", "NDCG@20",
            "MRR@10", "MRR@20",
            "all_equal_ratio", "tie_case_ratio", "avg_tie_items",
            "mean_rank_1based",
        ):
            if key in metrics:
                row[key] = metrics[key]

        _append_csv(result_dir / "unisrec_zero_shot.csv", row)
        all_results.append(row)

        elapsed_s = float(time.perf_counter() - target_start)
        timings.append({
            "label": target,
            "elapsed_seconds": elapsed_s,
            "examples": int(metrics.get("num_examples", 0)),
        })

        log_event("EVAL", "TARGET_DONE", run=run_name, target=target,
                  examples=int(metrics.get("num_examples", 0)),
                  ndcg10=float(metrics.get("NDCG@10", float("nan"))),
                  recall10=float(metrics.get("Recall@10", float("nan"))),
                  elapsed=format_seconds(elapsed_s))
        maybe_print_json(row, enabled=bool(args.print_json))

    # ── Save results ───────────────────────────────────────────────
    output_path = (
        result_dir / f"{run_name}_{source}_seed{seed}_zero_shot.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results": all_results,
                "runtime": {
                    "show_progress": show_progress,
                    "print_json": args.print_json,
                    "progress_refresh_seconds": progress_refresh,
                    "leave_progress_bar": bool(
                        rt_cfg.get("leave_progress_bar", True)
                    ),
                    "run_elapsed_seconds": float(clock.elapsed()),
                    "targets": timings,
                },
            },
            f, ensure_ascii=False, indent=2,
        )

    chart_path = save_stage_timing_chart(
        timings,
        result_dir / f"{run_name}_{source}_seed{seed}_zero_shot_runtime.png",
        title=f"{run_name}: zero-shot evaluation time",
    )

    log_event("ARTIFACT", "SAVED", results=str(output_path))
    if chart_path:
        log_event("ARTIFACT", "TIMING_CHART", path=str(chart_path))
    log_event("RUN", "DONE", stage="unisrec_zero_shot",
              run=run_name, elapsed=format_seconds(clock.elapsed()))


if __name__ == "__main__":
    main()
