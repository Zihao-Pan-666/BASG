from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from basg.data.dataset import collate_prefix
from basg.evaluation.bert4rec_family_evaluator import evaluate_bert4rec_family
from basg.features.semantic import load_semantic_embeddings
from basg.models.bert4rec_family import BERT4RecSemanticFamily
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
    resolve_runtime_config,
    save_stage_timing_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final zero-shot evaluation of a source-selected family checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override data.seed and update seed-dependent paths")
    parser.add_argument("--targets", default=None)
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
            header = next(csv.reader(handle), None)
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

    if args.seed is not None:
        _apply_seed_override(cfg, int(args.seed))

    runtime = resolve_runtime_config(
        cfg.get("runtime"),
        no_progress=args.no_progress,
        print_json=args.print_json,
        progress_refresh=args.progress_refresh,
    )
    clock = RunClock()

    experiment = cfg["experiment"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    evaluation_cfg = dict(cfg["evaluation"])
    source = str(cfg["source"])
    declared_targets = [str(x) for x in cfg["targets"]]
    targets = (
        [target.strip() for target in args.targets.split(",") if target.strip()]
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

    default_checkpoint = (
        Path(cfg["paths"]["checkpoint_dir"]) / f"{run_name}_{source}_seed{seed}.pt"
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    set_seed_compat(seed)
    device = resolve_device_compat(str(training_cfg.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint.get("source_domain") != source:
        raise RuntimeError(
            f"Checkpoint source={checkpoint.get('source_domain')} but config source={source}"
        )
    if checkpoint.get("best_metric_name") != evaluation_cfg["primary_metric"]:
        raise RuntimeError("Checkpoint selection metric differs from config")

    result_dir = Path(cfg["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "RUN",
        "START",
        stage="baseline_zero_shot",
        run=run_name,
        source=source,
        targets=targets,
        seed=seed,
        device=device,
    )

    all_results: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        target_started = time.perf_counter()
        log_event(
            "EVAL",
            "TARGET_START",
            run=run_name,
            target=target,
            target_index=f"{target_index + 1}/{len(targets)}",
        )
        interactions, user_map, item_map = load_interactions_compat(data_root, target)
        sequences = group_user_sequences_compat(
            interactions, min_len=int(data_cfg["min_len"])
        )
        samples = build_target_samples_compat(
            sequences=sequences,
            max_len=int(data_cfg["max_len"]),
            min_len=int(data_cfg["min_len"]),
        )
        item_features = load_semantic_embeddings(
            data_root=data_root,
            domain=target,
            item_map=item_map,
            embedding_dir=embedding_dir,
        )

        hparams = checkpoint["model_hparams"]
        model = BERT4RecSemanticFamily(
            item_features=item_features,
            hidden_dim=int(hparams["hidden_dim"]),
            max_len=int(hparams["max_len"]),
            num_layers=int(hparams["num_layers"]),
            num_heads=int(hparams["num_heads"]),
            dropout=float(hparams["dropout"]),
            architecture=str(hparams["architecture"]),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        loader = DataLoader(
            make_prefix_dataset(
                samples,
                num_items=len(item_map),
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
            + target_index,
            max_batches=int(evaluation_cfg.get("max_batches", 0)),
            show_progress=bool(runtime["show_progress"]),
            progress_desc=f"{run_name}:{target}",
            progress_refresh=float(runtime["progress_refresh_seconds"]),
            leave_progress=bool(runtime["leave_progress_bar"]),
        )

        row: dict[str, object] = {
            "run_name": checkpoint["run_name"],
            "mode": checkpoint["mode"],
            "architecture": hparams["architecture"],
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
            "Recall@10",
            "Recall@20",
            "NDCG@10",
            "NDCG@20",
            "MRR@10",
            "MRR@20",
            "all_equal_ratio",
            "tie_case_ratio",
            "avg_tie_items",
            "mean_rank_1based",
        ):
            if key in metrics:
                row[key] = metrics[key]

        append_csv(result_dir / "bert4rec_family_zero_shot.csv", row)
        all_results.append(row)
        elapsed_seconds = float(time.perf_counter() - target_started)
        timings.append(
            {
                "label": target,
                "elapsed_seconds": elapsed_seconds,
                "examples": int(metrics.get("num_examples", 0)),
                "batches": int(metrics.get("num_batches", 0)),
            }
        )
        log_event(
            "EVAL",
            "TARGET_DONE",
            run=run_name,
            target=target,
            examples=int(metrics.get("num_examples", 0)),
            ndcg10=float(metrics.get("NDCG@10", float("nan"))),
            recall10=float(metrics.get("Recall@10", float("nan"))),
            elapsed=format_seconds(elapsed_seconds),
        )
        maybe_print_json(row, enabled=bool(runtime["print_json"]))

    output_path = result_dir / f"{checkpoint['run_name']}_{source}_seed{seed}_zero_shot.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results": all_results,
                "runtime": {
                    **runtime,
                    "run_elapsed_seconds": float(clock.elapsed()),
                    "targets": timings,
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    chart_path = save_stage_timing_chart(
        timings,
        result_dir / f"{checkpoint['run_name']}_{source}_seed{seed}_zero_shot_runtime.png",
        title=f"{checkpoint['run_name']}: zero-shot evaluation time",
    )
    log_event("ARTIFACT", "SAVED", results=output_path)
    if chart_path:
        log_event("ARTIFACT", "TIMING_CHART", path=chart_path)
    log_event(
        "RUN",
        "DONE",
        stage="baseline_zero_shot",
        run=run_name,
        elapsed=format_seconds(clock.elapsed()),
    )


if __name__ == "__main__":
    main()
