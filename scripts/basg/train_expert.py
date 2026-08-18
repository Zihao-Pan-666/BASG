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
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from basg.data.popdyn_dataset import PopDynPrefixDataset, collate_popdyn_prefix
from basg.features.temporal import PREFIX_TEMPORAL_FEATURE_NAMES, build_prefix_temporal_features
from basg.features.semantic import load_semantic_embeddings
from basg.models.temporal_expert import PopDynSequentialExpert
from basg.training.temporal_trainer import train_popdyn_expert
from basg.utils.bert4rec_family_compat import (
    build_source_splits_compat,
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
    save_training_timing_chart,
)
from basg.utils.source_partition import partition_validation_samples_by_user


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing != fieldnames:
            raise RuntimeError(
                f"CSV schema mismatch for {path}; use a fresh strict result directory."
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
    parser = argparse.ArgumentParser(description="Train source-only temporal behavior expert.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override data.seed and update seed-dependent paths")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-progress", "--no_progress", action="store_true", dest="no_progress")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--progress-refresh", type=float, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None,
                        help="Override temporal_training.max_batches_per_epoch for fast smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.epochs is not None:
        cfg["temporal_training"]["epochs"] = int(args.epochs)
    if args.run_name is not None:
        cfg["experiment"]["temporal_run_name"] = str(args.run_name)
    if args.seed is not None:
        _apply_seed_override(cfg, int(args.seed))
    if args.max_batches_per_epoch is not None:
        cfg["temporal_training"]["max_batches_per_epoch"] = int(args.max_batches_per_epoch)
        log_event("CLI", "MAX_BATCHES", max_batches_per_epoch=int(args.max_batches_per_epoch))
    runtime = resolve_runtime_config(
        cfg.get("runtime"),
        no_progress=args.no_progress,
        print_json=args.print_json,
        progress_refresh=args.progress_refresh,
    )
    clock = RunClock()

    data_cfg = cfg["data"]
    source = str(cfg["source"])
    data_root = str(cfg.get("data_root", "./data"))
    seed = int(data_cfg.get("seed", 2026))
    set_seed_compat(seed)
    device = resolve_device_compat(str(cfg["temporal_training"].get("device", "auto")))
    run_name = str(cfg["experiment"]["temporal_run_name"])
    log_event(
        "RUN",
        "START",
        stage="temporal_expert_train",
        run=run_name,
        source=source,
        seed=seed,
        device=device,
    )

    # Only the declared source is loaded in this training script.
    interactions, _, item_map = load_interactions_compat(data_root, source)
    sequences = group_user_sequences_compat(
        interactions, min_len=int(data_cfg["min_len"])
    )
    train_samples, val_samples = build_source_splits_compat(
        sequences,
        max_len=int(data_cfg["max_len"]),
        min_len=int(data_cfg["min_len"]),
    )
    popdyn_selection, fusion_selection, router_calibration = (
        partition_validation_samples_by_user(
            val_samples,
            popdyn_selection_fraction=float(
                cfg["source_partitions"]["popdyn_selection_user_fraction"]
            ),
            fusion_selection_fraction=float(
                cfg["source_partitions"]["fusion_selection_user_fraction"]
            ),
            salt=seed + int(cfg["source_partitions"].get("split_salt", 31415)),
        )
    )
    item_features = load_semantic_embeddings(
        data_root=data_root,
        domain=source,
        item_map=item_map,
        embedding_dir=str(data_cfg.get("embedding_dir", "semantic_embeddings")),
    )

    def loader(
        samples: list[dict],
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        return DataLoader(
            PopDynPrefixDataset(
                samples, num_items=len(item_map), max_len=int(data_cfg["max_len"])
            ),
            batch_size=int(cfg["temporal_training"]["batch_size"]),
            shuffle=shuffle,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_popdyn_prefix,
            pin_memory=device.type == "cuda",
            generator=generator,
        )

    log_event(
        "DATA",
        "READY",
        source=source,
        items=len(item_map),
        train_samples=len(train_samples),
        popdyn_selection=len(popdyn_selection),
        fusion_selection=len(fusion_selection),
        router_calibration=len(router_calibration),
    )

    model_cfg = cfg["temporal_model"]
    # Compute feature_dim: 2 base temporal + 1 popularity (if predictor available)
    feature_dim = len(PREFIX_TEMPORAL_FEATURE_NAMES)  # currently 3

    # ── Build popularity lookup from LLM→pop predictor ──────────────
    pop_lookup: torch.Tensor | None = None
    pop_predictor_path = cfg["temporal_training"].get("pop_predictor_path")
    if pop_predictor_path and Path(pop_predictor_path).exists():
        from basg.features.popularity import build_pop_lookup
        pop_lookup = build_pop_lookup(
            item_features,
            pop_predictor_path,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        log_event("DATA", "POP_LOOKUP", shape=tuple(pop_lookup.shape),
                  min=float(pop_lookup.min().item()), max=float(pop_lookup.max().item()))
    else:
        log_event("DATA", "POP_LOOKUP", status="skipped")

    # ── Build behavior-augmented lookup (v8 AlphaFree) ──────────────
    behavior_lookup: torch.Tensor | None = None
    behavior_dim = 0
    behavior_mapper_path = cfg["temporal_training"].get("behavior_mapper_path")
    if behavior_mapper_path and Path(behavior_mapper_path).exists():
        from basg.features.behavior import build_behavior_lookup
        behavior_lookup = build_behavior_lookup(
            item_features,
            behavior_mapper_path,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        behavior_dim = behavior_lookup.shape[1]
        log_event("DATA", "BEHAVIOR_LOOKUP", shape=tuple(behavior_lookup.shape),
                  behavior_lookup_enabled=True,
                  candidate_encoder_input_dim=4096 + behavior_dim)
    else:
        log_event("DATA", "BEHAVIOR_LOOKUP", status="skipped")

    model = PopDynSequentialExpert(
        feature_dim=feature_dim,
        semantic_dim=int(item_features.shape[1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        max_len=int(data_cfg["max_len"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        position_mode=str(model_cfg.get("position_mode", "fixed")),
        behavior_dim=behavior_dim,
        behavior_gate_init=float(cfg["temporal_training"].get("behavior_gate_init", 0.1)),
        fail_fast_behavior=bool(cfg["temporal_training"].get("fail_fast_behavior", True)),
    )
    # v8 asserts
    assert model.semantic_dim == 4096, f"semantic_dim={model.semantic_dim}, expected 4096"
    checkpoint = Path(cfg["paths"]["temporal_checkpoint"])
    eval_cfg = {
        **dict(cfg["evaluation"]),
        "ranking": str(data_cfg["ranking"]),
        "eval_negatives": int(data_cfg["eval_negatives"]),
    }
    train_cfg = {
        **dict(cfg["temporal_training"]),
        "train_negatives": int(data_cfg["train_negatives"]),
        "runtime": runtime,
        "pop_lookup": pop_lookup,
        "behavior_lookup": behavior_lookup,
        "ablation": dict(cfg.get("ablation", {}) or {}),
    }
    # ── Smoke check: single-batch forward pass before training ──
    model.to(device)
    ablation_smoke = dict(cfg.get("ablation", {}) or {})
    ut_smoke = bool(ablation_smoke.get("use_temporal_features", True))
    uip_smoke = bool(ablation_smoke.get("use_interaction_property", True))
    with torch.no_grad():
        smoke_loader = loader(train_samples, shuffle=False)
        smoke_batch = next(iter(smoke_loader))
        smoke_h_ids = smoke_batch["history"][:4].to(device)
        smoke_h_times = smoke_batch["history_times"][:4]
        smoke_h_feats = build_prefix_temporal_features(
            smoke_h_ids.cpu(),
            smoke_h_times,
            time_scale_ms=float(cfg["temporal_features"]["time_scale_ms"]),
            pop_lookup=pop_lookup,
            use_temporal_features=ut_smoke,
            use_interaction_property=uip_smoke,
        ).to(device)
        smoke_c_ids = torch.randint(1, len(item_map) + 1, (4, 5))
        smoke_c_sem = item_features[smoke_c_ids].to(device)
        smoke_beh_dev = behavior_lookup.to(device) if behavior_lookup is not None else None
        smoke_scores = model.score_candidates(smoke_h_ids, smoke_h_feats, smoke_c_sem,
                                               smoke_c_ids.to(device), smoke_beh_dev,
                                               use_interaction_property=uip_smoke)
        if not torch.isfinite(smoke_scores).all():
            raise FloatingPointError("Smoke forward produced NaN/Inf scores before training.")
        log_event("SMOKE", "PASS", scores_shape=tuple(smoke_scores.shape),
                  score_min=float(smoke_scores.min().item()),
                  score_max=float(smoke_scores.max().item()))
    log_event("SMOKE", "DONE")

    summary = train_popdyn_expert(
        model=model,
        semantic_item_features=item_features,
        train_loader=loader(
            train_samples,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 401),
        ),
        selection_loader=loader(popdyn_selection, shuffle=False),
        device=device,
        num_items=len(item_map),
        checkpoint_path=checkpoint,
        training_cfg=train_cfg,
        evaluation_cfg=eval_cfg,
        temporal_cfg=dict(cfg["temporal_features"]),
        seed=seed,
        source_domain=source,
        run_name=run_name,
    )
    result_dir = Path(cfg["paths"]["result_dir"])
    chart_path = None
    if runtime["save_timing_chart"]:
        chart_path = save_training_timing_chart(
            summary["history"],
            result_dir / f"{run_name}_{source}_seed{seed}_runtime.png",
            title=f"{run_name}: training time",
        )
    summary.update(
        {
            "num_train_samples": len(train_samples),
            "num_popdyn_selection_samples": len(popdyn_selection),
            "num_fusion_selection_samples": len(fusion_selection),
            "num_router_calibration_samples": len(router_calibration),
            "source_validation_protocol": (
                "three_disjoint_user_partitions: checkpoint_selection, "
                "static_beta_selection, router_calibration"
            ),
            "runtime": {
                **runtime,
                "run_elapsed_seconds": float(clock.elapsed()),
                "timing_chart_path": chart_path,
            },
            "strict_zero_shot_protocol": {
                "target_interactions_loaded_during_training": False,
                "target_catalog_statistics_used": False,
            },
        }
    )
    output = result_dir / f"{run_name}_{source}_seed{seed}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_csv(
        result_dir / "temporal_expert_source_val.csv",
        {
            "run_name": run_name,
            "source": source,
            "seed": seed,
            "best_epoch": summary["best_epoch"],
            "selection_metric": summary["best_metric_name"],
            "source_val_metric": summary["best_metric_value"],
            "checkpoint_path": summary["checkpoint_path"],
            "summary_path": str(output),
            "target_data_used_in_training": False,
            "target_statistics_used_in_training": False,
        },
    )
    log_event("ARTIFACT", "SAVED", checkpoint=checkpoint, summary=output)
    if chart_path:
        log_event("ARTIFACT", "TIMING_CHART", path=chart_path)
    log_event(
        "RUN",
        "DONE",
        stage="temporal_expert_train",
        run=run_name,
        best_epoch=summary["best_epoch"],
        metric=f"{summary['best_metric_name']}:{summary['best_metric_value']:.6f}",
        elapsed=format_seconds(clock.elapsed()),
    )
    maybe_print_json(summary, enabled=bool(runtime["print_json"]))


if __name__ == "__main__":
    main()
