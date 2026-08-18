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

from basg.data.popdyn_dataset import PopDynPrefixDataset, collate_popdyn_prefix
from basg.evaluation.dual_expert_evaluator import evaluate_dual_experts
from basg.features.semantic import load_semantic_embeddings
from basg.models.family_checkpoint import load_semantic_backbone
from basg.models.temporal_expert import PopDynSequentialExpert
from basg.utils.bert4rec_family_compat import (
    build_target_samples_compat,
    group_user_sequences_compat,
    load_interactions_compat,
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


def _torch_load(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_behavior_expert(
    path: str | Path, device: torch.device
) -> tuple[PopDynSequentialExpert, dict]:
    payload = _torch_load(path)
    hp = payload["model_hparams"]
    model = PopDynSequentialExpert(
        feature_dim=int(hp["feature_dim"]),
        semantic_dim=int(hp["semantic_dim"]),
        hidden_dim=int(hp["hidden_dim"]),
        max_len=int(hp["max_len"]),
        num_layers=int(hp["num_layers"]),
        num_heads=int(hp["num_heads"]),
        dropout=float(hp["dropout"]),
        position_mode=str(hp.get("position_mode", "fixed")),
        behavior_dim=int(hp.get("behavior_dim", 0)),
        behavior_gate_init=float(hp.get("behavior_gate", 0.1)),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def append_csv(path: Path, row: dict[str, object]) -> None:
    """Append one result row with a tolerant, stable CSV schema.

    Why this is needed:
    evaluate_dual_experts may return diagnostic tie fields for some methods
    but not for others, for example:
      all_equal_ratio, tie_case_ratio, avg_tie_items

    Therefore, a valid run can produce rows with different key sets inside the
    same target/method loop. The old strict comparison incorrectly crashed after
    the first row created the header.

    Policy:
      1. New or empty file: write the current row schema as header.
      2. Existing file with all needed columns: append and fill missing cells.
      3. Existing file missing new columns: rewrite the CSV with a union header,
         preserving old column order and appending new columns at the end.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming_names = list(row.keys())

    exists = path.exists() and path.stat().st_size > 0
    if not exists:
        fieldnames = incoming_names
        old_rows: list[dict[str, object]] = []
        rewrite_existing = False
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_names = list(reader.fieldnames or [])
            old_rows = list(reader)

        if not existing_names:
            fieldnames = incoming_names
            rewrite_existing = True
        else:
            extra_in_new = [name for name in incoming_names if name not in existing_names]
            fieldnames = existing_names + extra_in_new
            rewrite_existing = bool(extra_in_new)

    if rewrite_existing:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for old_row in old_rows:
                writer.writerow({name: old_row.get(name, "") for name in fieldnames})

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _apply_seed_override(cfg: dict, new_seed: int) -> None:
    """Override data.seed and replace seed in all path values."""
    old = str(cfg["data"].get("seed", ""))
    cfg["data"]["seed"] = int(new_seed)
    if old:
        for key, value in cfg.get("paths", {}).items():
            if isinstance(value, str):
                cfg["paths"][key] = re.sub(rf"seed{old}", f"seed{new_seed}", value)
    # Also resolve {seed} template in all path values
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str):
            cfg["paths"][key] = value.replace("{seed}", str(new_seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict zero-shot evaluation of frozen backbone + Behavior Expert."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override data.seed and update seed-dependent paths")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--no-progress", "--no_progress", action="store_true", dest="no_progress")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--progress-refresh", type=float, default=None)
    parser.add_argument(
        "--ablation",
        default="none",
        choices=["none", "w/o temporal-features", "w/o item-properties"],
        help="Ablation variant: 'w/o temporal-features' zeros gap+recency (keeps pop+cooc); "
             "'w/o item-properties' zeros pop+cooc (keeps gap+recency).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.seed is not None:
        _apply_seed_override(cfg, int(args.seed))

    ablation: str | None = args.ablation if args.ablation != "none" else None

    runtime = resolve_runtime_config(
        cfg.get("runtime"),
        no_progress=args.no_progress,
        print_json=args.print_json,
        progress_refresh=args.progress_refresh,
    )
    clock = RunClock()
    data_cfg = cfg["data"]
    source = str(cfg["source"])
    all_targets = [str(x) for x in cfg["targets"]]
    targets = (
        [x.strip() for x in args.targets.split(",") if x.strip()]
        if args.targets
        else all_targets
    )
    unknown = set(targets) - set(all_targets)
    if unknown:
        raise ValueError(f"Targets not declared in config: {sorted(unknown)}")

    seed = int(data_cfg.get("seed", 2026))
    data_root = str(cfg.get("data_root", "./data"))
    set_seed_compat(seed)
    device = resolve_device_compat(str(cfg["temporal_training"].get("device", "auto")))
    fusion = _torch_load(cfg["paths"]["fusion_checkpoint"])
    behavior_model, behavior_payload = _load_behavior_expert(
        cfg["paths"]["temporal_checkpoint"], device
    )
    result_dir = Path(cfg["paths"]["result_dir"])
    run_name = str(cfg["experiment"]["fusion_run_name"])
    if ablation:
        result_dir = Path("results/ablation")
        run_name = f"{run_name}_{ablation.replace(' ', '_').replace('/', '_')}"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_event(
        "RUN",
        "START",
        stage="strict_zero_shot",
        run=run_name,
        source=source,
        targets=targets,
        seed=seed,
        device=device,
    )

    all_rows: list[dict] = []
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
        # Target interactions are read only to construct each user's observed
        # prefix and held-out next-item evaluation example. No target-wide
        # aggregation is computed.
        interactions, _, item_map = load_interactions_compat(data_root, target)
        sequences = group_user_sequences_compat(
            interactions, min_len=int(data_cfg["min_len"])
        )
        samples = build_target_samples_compat(
            sequences,
            max_len=int(data_cfg["max_len"]),
            min_len=int(data_cfg["min_len"]),
        )
        item_features = load_semantic_embeddings(
            data_root=data_root,
            domain=target,
            item_map=item_map,
            embedding_dir=str(data_cfg.get("embedding_dir", "semantic_embeddings")),
        )

        # ── Build source popularity lookup for target domain ──
        pop_lookup: torch.Tensor | None = None
        pop_predictor_path = cfg["temporal_training"].get("pop_predictor_path")
        if pop_predictor_path and Path(pop_predictor_path).exists():
            from basg.features.popularity import build_pop_lookup
            pop_lookup = build_pop_lookup(item_features, pop_predictor_path, device)
            log_event("DATA", "POP_LOOKUP", target=target,
                      shape=tuple(pop_lookup.shape),
                      min=float(pop_lookup.min().item()),
                      max=float(pop_lookup.max().item()))
        else:
            log_event("DATA", "POP_LOOKUP", target=target, status="skipped")

        # ── Build behavior lookup for target domain ──
        behavior_lookup: torch.Tensor | None = None
        behavior_mapper_path = cfg["temporal_training"].get("behavior_mapper_path")
        if behavior_mapper_path and Path(behavior_mapper_path).exists():
            from basg.features.behavior import build_behavior_lookup
            behavior_lookup = build_behavior_lookup(item_features, behavior_mapper_path, device)
            log_event("DATA", "BEHAVIOR_LOOKUP", target=target,
                      shape=tuple(behavior_lookup.shape))
        else:
            log_event("DATA", "BEHAVIOR_LOOKUP", target=target, status="skipped")

        # ── Ablation: w/o item-properties → zero pop_lookup & behavior_lookup ──
        if ablation == "w/o item-properties":
            if pop_lookup is not None:
                pop_lookup = torch.zeros_like(pop_lookup)
            if behavior_lookup is not None:
                behavior_lookup = torch.zeros_like(behavior_lookup)
            log_event("ABLATION", "WO_ITEM_PROPERTIES", target=target,
                      pop_zeroed=pop_lookup is not None,
                      behavior_zeroed=behavior_lookup is not None)

        semantic_model, semantic_payload = load_semantic_backbone(
            cfg["paths"]["backbone_checkpoint"],
            item_features,
            device,
        )
        if behavior_model.semantic_dim != int(item_features.shape[1]):
            raise RuntimeError(
                "Target semantic embedding dimension differs from temporal-expert checkpoint."
            )

        loader = DataLoader(
            PopDynPrefixDataset(samples, len(item_map), int(data_cfg["max_len"])),
            batch_size=int(cfg["temporal_training"]["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_popdyn_prefix,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_dual_experts(
            semantic_model=semantic_model,
            behavior_model=behavior_model,
            semantic_item_features=item_features,
            loader=loader,
            device=device,
            num_items=len(item_map),
            ranking=str(data_cfg["ranking"]),
            eval_negatives=int(data_cfg["eval_negatives"]),
            ks=tuple(int(x) for x in cfg["evaluation"].get("ks", [10, 20])),
            tie_policy=str(cfg["evaluation"].get("tie_policy", "worst")),
            seed=seed
            + int(cfg["evaluation"].get("target_seed_offset", 20000))
            + target_index,
            static_beta=float(fusion["static_beta"]),
            time_scale_ms=float(
                cfg["temporal_features"].get("time_scale_ms", 86_400_000.0)
            ),
            max_batches=int(cfg["evaluation"].get("max_batches", 0)),
            pop_lookup=pop_lookup,
            behavior_lookup=behavior_lookup,
            ablation=ablation,
            show_progress=bool(runtime["show_progress"]),
            progress_desc=f"{run_name}:{target}",
            progress_refresh=float(runtime["progress_refresh_seconds"]),
            leave_progress=bool(runtime["leave_progress_bar"]),
        )

        target_elapsed = float(time.perf_counter() - target_started)
        timings.append(
            {
                "label": target,
                "elapsed_seconds": target_elapsed,
                "examples": int(metrics["Backbone"].get("num_examples", 0)),
            }
        )
        for method, values in metrics.items():
            row = {
                "run_name": run_name,
                "method": method,
                "ablation": ablation if ablation else "none",
                "source": source,
                "target": target,
                "seed": seed,
                "source_backbone_best_epoch": semantic_payload.get("best_epoch"),
                "source_temporal_best_epoch": behavior_payload.get("best_epoch"),
                "source_static_beta": float(fusion["static_beta"]),
                "ranking": str(data_cfg["ranking"]),
                "eval_negatives": int(data_cfg["eval_negatives"]),
                "target_interactions_used_for_training": False,
                "target_interactions_used_for_model_selection": False,
                "target_catalog_interaction_statistics_used": False,
                "target_prefix_timestamps_used_as_inference_input": True,
                "target_item_semantic_embeddings_used": True,
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"elapsed_seconds", "num_batches"}
                },
            }
            append_csv(result_dir / "zero_shot_eval.csv", row)
            all_rows.append(row)
            log_event(
                "EVAL",
                "RESULT",
                run=run_name,
                target=target,
                method=method,
                ndcg10=float(values.get("NDCG@10", float("nan"))),
                recall10=float(values.get("Recall@10", float("nan"))),
                mean_rank=float(values.get("mean_rank_1based", float("nan"))),
            )
            maybe_print_json(row, enabled=bool(runtime["print_json"]))

        log_event(
            "EVAL",
            "TARGET_DONE",
            run=run_name,
            target=target,
            examples=int(metrics["Backbone"].get("num_examples", 0)),
            elapsed=format_seconds(target_elapsed),
        )

    output = result_dir / f"{run_name}_{source}_seed{seed}_zero_shot.json"
    output.write_text(
        json.dumps(
            {
                "results": all_rows,
                "runtime": {
                    **runtime,
                    "run_elapsed_seconds": float(clock.elapsed()),
                    "targets": timings,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    chart_path = None
    if runtime["save_timing_chart"]:
        chart_path = save_stage_timing_chart(
            timings,
            result_dir / f"{run_name}_{source}_seed{seed}_runtime.png",
            title=f"{run_name}: zero-shot runtime",
        )
    log_event("ARTIFACT", "SAVED", results=output)
    if chart_path:
        log_event("ARTIFACT", "TIMING_CHART", path=chart_path)
    log_event(
        "RUN",
        "DONE",
        stage="strict_zero_shot",
        run=run_name,
        elapsed=format_seconds(clock.elapsed()),
    )


if __name__ == "__main__":
    main()
