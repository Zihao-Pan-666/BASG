"""BASG dual-expert evaluation: semantic backbone, Behavior Expert, Static Fusion.

Provides the two evaluators used by the BASG pipeline:

- ``evaluate_popdyn_expert``: sampled-ranking evaluation of the standalone
  Behavior Expert (used for source-side checkpoint selection during training).
- ``evaluate_dual_experts``: paired evaluation of the frozen semantic
  backbone, the Behavior Expert, and their query-wise standardized Static
  Fusion (Eq. (7) in the paper). An oracle diagnostic reports the best
  achievable rank when either expert alone is correct.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from basg.evaluation.base import (
    build_candidate_batch,
    metrics_from_ranks,
    ranks_from_scores,
    score_diagnostics,
    zscore_rows,
)
from basg.features.temporal import build_prefix_temporal_features
from basg.utils.runtime import (
    RunClock,
    close_progress,
    format_seconds,
    loader_total,
    progress_iter,
    resolve_progress_enabled,
    set_progress_postfix,
)


@torch.no_grad()
def evaluate_popdyn_expert(
    *,
    model: torch.nn.Module,
    semantic_item_features: torch.Tensor,
    loader: Iterable,
    device: torch.device,
    num_items: int,
    ranking: str,
    eval_negatives: int,
    ks: tuple[int, ...],
    tie_policy: str,
    seed: int,
    time_scale_ms: float,
    max_batches: int = 0,
    show_progress: bool | None = None,
    progress_desc: str = "behavior_expert",
    progress_refresh: float = 0.5,
    leave_progress: bool = False,
    pop_lookup: torch.Tensor | np.ndarray | None = None,
    behavior_lookup: torch.Tensor | np.ndarray | None = None,
    use_temporal_features: bool = True,
    use_interaction_property: bool = True,
) -> dict[str, float]:
    """Evaluate a standalone Behavior Expert on sampled ranking."""
    if ranking != "sampled":
        raise ValueError("Temporal expert evaluation supports sampled ranking only.")
    model.eval()
    semantic_table_cpu = semantic_item_features.detach().float().cpu()
    rng = np.random.default_rng(int(seed))
    ranks: list[int] = []
    score_chunks: list[np.ndarray] = []
    clock = RunClock()
    iterator = progress_iter(
        loader,
        total=loader_total(loader, max_batches),
        run_name=progress_desc,
        phase="EVAL",
        detail="behavior_expert",
        enabled=resolve_progress_enabled(show_progress),
        refresh_seconds=progress_refresh,
        leave=leave_progress,
    )
    try:
        for batch_index, batch in enumerate(iterator):
            if max_batches > 0 and batch_index >= max_batches:
                break
            history_ids_cpu = batch["history"]
            history_times_cpu = batch["history_times"]

            candidates_np, target_indices = build_candidate_batch(
                history_ids_cpu.tolist(),
                batch["target"].tolist(),
                num_items=int(num_items),
                eval_negatives=int(eval_negatives),
                rng=rng,
            )
            history_ids = history_ids_cpu.to(device, non_blocking=True)
            valid_mask = history_ids.ne(0)
            empty_rows = int((~valid_mask.any(dim=1)).sum().item())
            if empty_rows > 0:
                import warnings
                warnings.warn(
                    f"[EVAL][EMPTY_PREFIX] batch={batch_index} empty_rows={empty_rows}"
                )
            temporal_features = build_prefix_temporal_features(
                history_ids_cpu,
                history_times_cpu,
                time_scale_ms=float(time_scale_ms),
                pop_lookup=pop_lookup,
                use_temporal_features=use_temporal_features,
                use_interaction_property=use_interaction_property,
            ).to(device, non_blocking=True)
            feat_absmax = float(temporal_features.abs().max().item())
            if not torch.isfinite(temporal_features).all():
                raise FloatingPointError(
                    f"Non-finite temporal_features at eval batch={batch_index}"
                )
            candidate_semantics = semantic_table_cpu[
                torch.as_tensor(candidates_np, dtype=torch.long)
            ].to(device, non_blocking=True)
            if not torch.isfinite(candidate_semantics).all():
                raise FloatingPointError(
                    f"Non-finite candidate_semantics at eval batch={batch_index}"
                )
            cand_ids_t = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
            scores = model.score_candidates(
                history_ids,
                temporal_features,
                candidate_semantics,
                cand_ids_t,
                behavior_lookup.to(device) if behavior_lookup is not None else None,
                use_interaction_property=use_interaction_property,
            ).detach().float().cpu().numpy()
            if not np.isfinite(scores).all():
                bad = ~np.isfinite(scores)
                bad_idx = np.argwhere(bad)
                first_bad = bad_idx[0].tolist() if bad_idx.size else None
                finite_scores = scores[np.isfinite(scores)]
                finite_min = float(finite_scores.min()) if finite_scores.size else float("nan")
                finite_max = float(finite_scores.max()) if finite_scores.size else float("nan")
                finite_absmax = float(np.abs(finite_scores).max()) if finite_scores.size else float("nan")
                raise FloatingPointError(
                    "Non-finite temporal-expert scores. "
                    f"batch={batch_index}, "
                    f"nan_count={int(np.isnan(scores).sum())}, "
                    f"inf_count={int(np.isinf(scores).sum())}, "
                    f"first_bad={first_bad}, "
                    f"finite_min={finite_min}, "
                    f"finite_max={finite_max}, "
                    f"finite_absmax={finite_absmax}, "
                    f"empty_rows={empty_rows}, "
                    f"feat_absmax={feat_absmax}"
                )
            ranks.extend(ranks_from_scores(scores, target_indices, tie_policy).tolist())
            score_chunks.append(scores)
            set_progress_postfix(
                iterator,
                examples=len(ranks),
                elapsed=format_seconds(clock.elapsed()),
            )
    finally:
        close_progress(iterator)

    if not ranks or not score_chunks:
        raise RuntimeError("Temporal-expert evaluation produced no examples.")
    metrics = metrics_from_ranks(np.asarray(ranks, dtype=np.int64), ks)
    metrics.update(score_diagnostics(np.concatenate(score_chunks, axis=0)))
    metrics["elapsed_seconds"] = float(clock.elapsed())
    metrics["num_batches"] = float(batch_index + 1 if "batch_index" in locals() else 0)
    return metrics


@torch.no_grad()
def evaluate_dual_experts(
    *,
    semantic_model: torch.nn.Module,
    behavior_model: torch.nn.Module,
    semantic_item_features: torch.Tensor,
    loader: Iterable,
    device: torch.device,
    num_items: int,
    ranking: str,
    eval_negatives: int,
    ks: tuple[int, ...],
    tie_policy: str,
    seed: int,
    static_beta: float,
    time_scale_ms: float,
    max_batches: int = 0,
    pop_lookup: torch.Tensor | np.ndarray | None = None,
    behavior_lookup: torch.Tensor | np.ndarray | None = None,
    ablation: str | None = None,
    use_temporal_features: bool = True,
    use_interaction_property: bool = True,
    show_progress: bool | None = None,
    progress_desc: str = "dual_expert",
    progress_refresh: float = 0.5,
    leave_progress: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate the semantic backbone, Behavior Expert, and Static Fusion.

    ``ablation`` accepts the inference-time ablations used for RQ2:
    ``"w/o temporal-features"`` zeroes the two timestamp-derived features and
    ``"w/o item-properties"`` zeroes the popularity and co-occurrence proxies
    (the latter is handled by the caller through zeroed lookups).
    """
    if ranking != "sampled":
        raise ValueError("Dual-expert fusion currently supports sampled ranking only.")
    semantic_model.eval()
    behavior_model.eval()
    semantic_table_cpu = semantic_item_features.detach().float().cpu()
    rng = np.random.default_rng(int(seed))

    feat_dim_expected = behavior_model.feature_dim
    _pop_enabled = pop_lookup is not None

    ranks_by_method: dict[str, list[int]] = {
        "Backbone": [],
        "BehaviorExpert": [],
        "StaticFusion": [],
        "OracleDiagnostic": [],
    }
    score_chunks: dict[str, list[np.ndarray]] = {
        "Backbone": [],
        "BehaviorExpert": [],
        "StaticFusion": [],
    }
    clock = RunClock()
    iterator = progress_iter(
        loader,
        total=loader_total(loader, max_batches),
        run_name=progress_desc,
        phase="EVAL",
        detail="dual_expert",
        enabled=resolve_progress_enabled(show_progress),
        refresh_seconds=progress_refresh,
        leave=leave_progress,
    )
    try:
        for batch_index, batch in enumerate(iterator):
            if max_batches > 0 and batch_index >= max_batches:
                break
            histories_cpu = batch["history"]
            history_times_cpu = batch["history_times"]
            targets_cpu = batch["target"]
            candidates_np, target_indices_np = build_candidate_batch(
                histories_cpu.tolist(),
                targets_cpu.tolist(),
                num_items=int(num_items),
                eval_negatives=int(eval_negatives),
                rng=rng,
            )
            histories = histories_cpu.to(device, non_blocking=True)
            _valid_dual = histories.ne(0)
            _empty_dual = int((~_valid_dual.any(dim=1)).sum().item())
            if _empty_dual > 0:
                import warnings
                warnings.warn(
                    f"[EVAL_DUAL][EMPTY_PREFIX] batch={batch_index} empty_rows={_empty_dual}"
                )
            candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
            temporal_features_cpu = build_prefix_temporal_features(
                histories_cpu,
                history_times_cpu,
                time_scale_ms=float(time_scale_ms),
                pop_lookup=pop_lookup,
                use_temporal_features=use_temporal_features,
                use_interaction_property=use_interaction_property,
            )
            # ── Inference-time ablation for non-retrained checkpoints ──
            if ablation == "w/o temporal-features" and use_temporal_features:
                temporal_features_cpu = temporal_features_cpu.clone()
                temporal_features_cpu[:, :, 0] = 0.0  # gap_log_normalized
                temporal_features_cpu[:, :, 1] = 0.0  # recency_fraction
                # column 2 (pop_percentile) retained
            temporal_features = temporal_features_cpu.to(device, non_blocking=True)
            if not torch.isfinite(temporal_features).all():
                raise FloatingPointError(
                    f"Non-finite temporal_features at evaluate_dual batch={batch_index}"
                )
            if temporal_features.shape[-1] != feat_dim_expected:
                raise ValueError(
                    f"temporal_features dim={temporal_features.shape[-1]} "
                    f"≠ model.feature_dim={feat_dim_expected}"
                )
            candidate_semantics = semantic_table_cpu[
                torch.as_tensor(candidates_np, dtype=torch.long)
            ].to(device, non_blocking=True)
            if not torch.isfinite(candidate_semantics).all():
                raise FloatingPointError(
                    f"Non-finite candidate_semantics at evaluate_dual batch={batch_index}"
                )

            cand_ids_eval = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
            beh_dev_eval = behavior_lookup.to(device) if behavior_lookup is not None else None
            semantic_scores = semantic_model.score_candidates(histories, candidates)
            behavior_scores = behavior_model.score_candidates(
                histories,
                temporal_features,
                candidate_semantics,
                cand_ids_eval,
                beh_dev_eval,
                use_interaction_property=use_interaction_property,
            )
            sem_np = semantic_scores.detach().float().cpu().numpy()
            beh_np = behavior_scores.detach().float().cpu().numpy()
            if not np.isfinite(sem_np).all() or not np.isfinite(beh_np).all():
                raise FloatingPointError(
                    "Non-finite backbone or temporal-expert scores at evaluate_dual. "
                    f"batch={batch_index}, "
                    f"sem_nan={int(np.isnan(sem_np).sum())}, "
                    f"sem_inf={int(np.isinf(sem_np).sum())}, "
                    f"beh_nan={int(np.isnan(beh_np).sum())}, "
                    f"beh_inf={int(np.isinf(beh_np).sum())}, "
                    f"empty_rows={_empty_dual}"
                )

            semantic_z = zscore_rows(sem_np)
            behavior_z = zscore_rows(beh_np)

            static_np = semantic_z + float(static_beta) * behavior_z

            score_map = {
                "Backbone": sem_np,
                "BehaviorExpert": beh_np,
                "StaticFusion": static_np,
            }
            rank_map = {
                key: ranks_from_scores(values, target_indices_np, tie_policy)
                for key, values in score_map.items()
            }
            rank_map["OracleDiagnostic"] = np.minimum(
                rank_map["Backbone"], rank_map["BehaviorExpert"]
            )
            for key, rank_values in rank_map.items():
                ranks_by_method[key].extend(rank_values.tolist())
            for key, score_values in score_map.items():
                score_chunks[key].append(score_values)

            set_progress_postfix(
                iterator,
                examples=len(ranks_by_method["Backbone"]),
                elapsed=format_seconds(clock.elapsed()),
            )
    finally:
        close_progress(iterator)

    if not ranks_by_method["Backbone"]:
        raise RuntimeError("Dual-expert evaluation produced no examples.")
    _CANONICAL_TIE_KEYS = ("all_equal_ratio", "tie_case_ratio", "avg_tie_items")

    metrics: dict[str, dict[str, float]] = {}
    for key, rank_values in ranks_by_method.items():
        values = metrics_from_ranks(np.asarray(rank_values, dtype=np.int64), ks)
        if key in score_chunks and score_chunks[key]:
            values.update(score_diagnostics(np.concatenate(score_chunks[key], axis=0)))
        else:
            for tie_key in _CANONICAL_TIE_KEYS:
                values.setdefault(tie_key, 0.0)
        values["elapsed_seconds"] = float(clock.elapsed())
        metrics[key] = values
    return metrics
