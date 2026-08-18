from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from basg.data.popdyn_dataset import PopDynPrefixDataset, collate_popdyn_prefix
from basg.evaluation.fusion_tuner import collect_dual_predictions, tune_static_beta
from basg.features.semantic import load_semantic_embeddings
from basg.models.family_checkpoint import load_semantic_backbone
from basg.models.temporal_expert import PopDynSequentialExpert
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
    save_stage_timing_chart,
)
from basg.utils.source_partition import partition_validation_samples_by_user


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


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
        description="Select source-only static fusion coefficient beta."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override data.seed and update seed-dependent paths")
    parser.add_argument("--no-progress", "--no_progress", action="store_true", dest="no_progress")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--progress-refresh", type=float, default=None)
    return parser.parse_args()


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

    data_cfg = cfg["data"]
    source = str(cfg["source"])
    data_root = str(cfg.get("data_root", "./data"))
    seed = int(data_cfg.get("seed", 2026))
    set_seed_compat(seed)
    device = resolve_device_compat(str(cfg["temporal_training"].get("device", "auto")))
    run_name = str(cfg["experiment"]["fusion_run_name"])
    log_event(
        "RUN",
        "START",
        stage="source_fusion_tuning",
        run=run_name,
        source=source,
        seed=seed,
        device=device,
    )

    interactions, _, item_map = load_interactions_compat(data_root, source)
    sequences = group_user_sequences_compat(
        interactions, min_len=int(data_cfg["min_len"])
    )
    _, val_samples = build_source_splits_compat(
        sequences,
        max_len=int(data_cfg["max_len"]),
        min_len=int(data_cfg["min_len"]),
    )
    _, fusion_selection, router_calibration = partition_validation_samples_by_user(
        val_samples,
        popdyn_selection_fraction=float(
            cfg["source_partitions"]["popdyn_selection_user_fraction"]
        ),
        fusion_selection_fraction=float(
            cfg["source_partitions"]["fusion_selection_user_fraction"]
        ),
        salt=seed + int(cfg["source_partitions"].get("split_salt", 31415)),
    )
    item_features = load_semantic_embeddings(
        data_root=data_root,
        domain=source,
        item_map=item_map,
        embedding_dir=str(data_cfg.get("embedding_dir", "semantic_embeddings")),
    )
    semantic_model, semantic_payload = load_semantic_backbone(
        cfg["paths"]["backbone_checkpoint"], item_features, device
    )
    behavior_model, behavior_payload = _load_behavior_expert(
        cfg["paths"]["temporal_checkpoint"], device
    )
    if behavior_model.semantic_dim != int(item_features.shape[1]):
        raise RuntimeError("Temporal checkpoint and source semantic embedding dimensions differ.")

    def make_loader(samples: list[dict]) -> DataLoader:
        return DataLoader(
            PopDynPrefixDataset(samples, len(item_map), int(data_cfg["max_len"])),
            batch_size=int(cfg["temporal_training"]["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_popdyn_prefix,
            pin_memory=device.type == "cuda",
        )

    # ── Build source popularity lookup (PrepRec-style) ──
    pop_lookup: torch.Tensor | None = None
    pop_predictor_path = cfg["temporal_training"].get("pop_predictor_path")
    if pop_predictor_path and Path(pop_predictor_path).exists():
        from basg.features.popularity import build_pop_lookup
        pop_lookup = build_pop_lookup(item_features, pop_predictor_path, device)
        log_event("DATA", "POP_LOOKUP", shape=tuple(pop_lookup.shape),
                  min=float(pop_lookup.min().item()), max=float(pop_lookup.max().item()))
    else:
        log_event("DATA", "POP_LOOKUP", status="skipped")

    # ── Build behavior-augmented lookup (AlphaFree-style) ──
    behavior_lookup: torch.Tensor | None = None
    behavior_mapper_path = cfg["temporal_training"].get("behavior_mapper_path")
    if behavior_mapper_path and Path(behavior_mapper_path).exists():
        from basg.features.behavior import build_behavior_lookup
        behavior_lookup = build_behavior_lookup(item_features, behavior_mapper_path, device)
        log_event("DATA", "BEHAVIOR_LOOKUP", shape=tuple(behavior_lookup.shape))
    else:
        log_event("DATA", "BEHAVIOR_LOOKUP", status="skipped")

    log_event(
        "DATA",
        "READY",
        source=source,
        items=len(item_map),
        fusion_selection=len(fusion_selection),
        router_calibration=len(router_calibration),
    )

    eval_seed = seed + int(cfg["evaluation"].get("fixed_negative_seed_offset", 10000))
    common = {
        "semantic_model": semantic_model,
        "behavior_model": behavior_model,
        "semantic_item_features": item_features,
        "device": device,
        "num_items": len(item_map),
        "ranking": str(data_cfg["ranking"]),
        "eval_negatives": int(data_cfg["eval_negatives"]),
        "time_scale_ms": float(
            cfg["temporal_features"].get("time_scale_ms", 86_400_000.0)
        ),
        "max_batches": int(cfg["evaluation"].get("max_batches", 0)),
        "show_progress": bool(runtime["show_progress"]),
        "progress_refresh": float(runtime["progress_refresh_seconds"]),
        "leave_progress": bool(runtime["leave_progress_bar"]),
        "pop_lookup": pop_lookup,
        "behavior_lookup": behavior_lookup,
    }
    timings: list[dict[str, object]] = []

    stage_started = time.perf_counter()
    selection_bundle = collect_dual_predictions(
        loader=make_loader(fusion_selection),
        seed=eval_seed,
        progress_desc=run_name,
        **common,
    )
    timings.append(
        {
            "label": "fusion_selection_scores",
            "elapsed_seconds": float(time.perf_counter() - stage_started),
        }
    )

    stage_started = time.perf_counter()
    selection_result = tune_static_beta(
        selection_bundle,
        beta_grid=[float(x) for x in cfg["fusion"]["static_beta_grid"]],
        tie_policy=str(cfg["evaluation"].get("tie_policy", "worst")),
        ks=tuple(int(x) for x in cfg["evaluation"].get("ks", [10, 20])),
        primary_metric=str(cfg["evaluation"]["primary_metric"]),
        show_progress=bool(runtime["show_progress"]),
        progress_desc=run_name,
        progress_refresh=float(runtime["progress_refresh_seconds"]),
        leave_progress=bool(runtime["leave_progress_bar"]),
    )
    timings.append(
        {
            "label": "static_beta_tuning",
            "elapsed_seconds": float(time.perf_counter() - stage_started),
        }
    )
    log_event(
        "TUNE",
        "STATIC_BETA",
        run=run_name,
        best_beta=float(selection_result["best_beta"]),
        metric=f"{cfg['evaluation']['primary_metric']}:{selection_result['best_metrics'][cfg['evaluation']['primary_metric']]:.6f}",
    )

    # Conservative beta: smallest beta within epsilon of the best source NDCG.
    best_ndcg = float(selection_result["best_metrics"][cfg["evaluation"]["primary_metric"]])
    epsilon = 0.001
    conservative_beta = float(selection_result["best_beta"])
    for trial in sorted(selection_result["trials"], key=lambda t: t["beta"]):
        if best_ndcg - float(trial[cfg["evaluation"]["primary_metric"]]) <= epsilon:
            conservative_beta = float(trial["beta"])
            break
    log_event("TUNE", "CONSERVATIVE_BETA", run=run_name,
              best_beta=float(selection_result["best_beta"]),
              conservative_beta=conservative_beta, epsilon=epsilon)

    fusion_payload = {
        "format_version": 6,
        "run_name": run_name,
        "source_domain": source,
        "seed": seed,
        "static_beta": float(selection_result["best_beta"]),
        "conservative_beta": conservative_beta,
        "selection_result": selection_result,
        "semantic_checkpoint": str(cfg["paths"]["backbone_checkpoint"]),
        "semantic_best_epoch": semantic_payload.get("best_epoch"),
        "temporal_checkpoint": str(cfg["paths"]["temporal_checkpoint"]),
        "temporal_best_epoch": behavior_payload.get("best_epoch"),
        "source_fusion_selection_examples": len(fusion_selection),
        "runtime": {
            **runtime,
            "run_elapsed_seconds": float(clock.elapsed()),
            "stages": timings,
        },
        "protocol": {
            "backbone_parameters_frozen": True,
            "fusion_weight_selected_on": "source_validation_fusion_selection_user_partition",
            "target_interactions_loaded_during_training_or_selection": False,
            "target_catalog_statistics_used": False,
        },
    }
    output = Path(cfg["paths"]["fusion_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fusion_payload, output)

    result_dir = Path(cfg["paths"]["result_dir"])
    summary = result_dir / f"{run_name}_{source}_seed{seed}_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(_json_safe(fusion_payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    chart_path = None
    if runtime["save_timing_chart"]:
        chart_path = save_stage_timing_chart(
            timings,
            result_dir / f"{run_name}_{source}_seed{seed}_runtime.png",
            title=f"{run_name}: fusion tuning runtime",
        )
    log_event("ARTIFACT", "SAVED", checkpoint=output, summary=summary)
    if chart_path:
        log_event("ARTIFACT", "TIMING_CHART", path=chart_path)
    log_event(
        "RUN",
        "DONE",
        stage="source_fusion_tuning",
        run=run_name,
        elapsed=format_seconds(clock.elapsed()),
    )
    maybe_print_json(fusion_payload, enabled=bool(runtime["print_json"]))


if __name__ == "__main__":
    main()
