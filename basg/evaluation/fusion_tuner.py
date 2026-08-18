"""Source-only fusion tuning: static beta search and dual-prediction collection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from basg.evaluation.base import (
    build_candidate_batch,
    metrics_from_ranks,
    ranks_from_scores,
    zscore_rows,
)
from basg.features.temporal import build_prefix_temporal_features
from basg.utils.runtime import (
    RunClock,
    close_progress,
    format_seconds,
    loader_total,
    progress_bar,
    progress_iter,
    resolve_progress_enabled,
    set_progress_postfix,
)


@dataclass
class DualPredictionBundle:
    """Paired semantic + behavior scores collected on one partition."""

    semantic_scores: np.ndarray
    behavior_scores: np.ndarray
    target_indices: np.ndarray

    def ranks(self, tie_policy: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            ranks_from_scores(self.semantic_scores, self.target_indices, tie_policy),
            ranks_from_scores(self.behavior_scores, self.target_indices, tie_policy),
        )


@torch.no_grad()
def collect_dual_predictions(
    *,
    semantic_model: torch.nn.Module,
    behavior_model: torch.nn.Module,
    semantic_item_features: torch.Tensor,
    loader: Iterable,
    device: torch.device,
    num_items: int,
    ranking: str = "sampled",
    eval_negatives: int = 100,
    seed: int = 12026,
    time_scale_ms: float = 86_400_000.0,
    max_batches: int = 0,
    show_progress: bool | None = None,
    progress_desc: str = "dual_predictions",
    progress_refresh: float = 0.5,
    leave_progress: bool = False,
    pop_lookup: torch.Tensor | np.ndarray | None = None,
    behavior_lookup: torch.Tensor | np.ndarray | None = None,
) -> DualPredictionBundle:
    """Collect paired semantic/behavior predictions on source validation.

    No popularity table or target-wide interaction statistic is accepted.
    On target, the only behavioral input is the current user's timestamped
    prefix — exactly as a sequential recommender is allowed to observe.
    """
    if ranking != "sampled":
        raise ValueError("The dual-expert package currently supports sampled ranking only.")
    semantic_model.eval()
    behavior_model.eval()
    semantic_table_cpu = semantic_item_features.detach().float().cpu()
    if semantic_table_cpu.shape[0] != num_items + 1:
        raise ValueError("Semantic table and num_items disagree.")
    rng = np.random.default_rng(int(seed))

    feat_dim_expected = behavior_model.feature_dim
    _pop_enabled = pop_lookup is not None

    semantic_chunks: list[np.ndarray] = []
    behavior_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    clock = RunClock()
    iterator = progress_iter(
        loader,
        total=loader_total(loader, max_batches),
        run_name=progress_desc,
        phase="EVAL",
        detail="paired_scores",
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
            candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
            histories = histories_cpu.to(device, non_blocking=True)

            _valid_col = histories_cpu.ne(0)
            _empty_col = int((~_valid_col.any(dim=1)).sum().item())
            if _empty_col > 0:
                import warnings
                warnings.warn(
                    f"[COLLECT_DUAL][EMPTY_PREFIX] batch={batch_index} empty_rows={_empty_col}"
                )

            temporal_features_cpu = build_prefix_temporal_features(
                histories_cpu,
                history_times_cpu,
                time_scale_ms=float(time_scale_ms),
                pop_lookup=pop_lookup,
            )
            temporal_features = temporal_features_cpu.to(device, non_blocking=True)
            if not torch.isfinite(temporal_features).all():
                raise FloatingPointError(
                    f"Non-finite temporal_features at dual-eval batch={batch_index}"
                )
            if temporal_features.shape[-1] != feat_dim_expected:
                raise ValueError(
                    f"temporal_features dim={temporal_features.shape[-1]} "
                    f"≠ model.feature_dim={feat_dim_expected}"
                )
            if batch_index == 0:
                import warnings
                warnings.warn(
                    f"[V7] temporal_features_dim={temporal_features.shape[-1]}, "
                    f"pop_lookup_enabled={_pop_enabled}"
                )

            candidate_semantics = semantic_table_cpu[
                torch.as_tensor(candidates_np, dtype=torch.long)
            ].to(device, non_blocking=True)
            if not torch.isfinite(candidate_semantics).all():
                raise FloatingPointError(
                    f"Non-finite candidate_semantics at dual-eval batch={batch_index}"
                )

            cand_ids_col = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
            beh_dev_col = behavior_lookup.to(device) if behavior_lookup is not None else None
            semantic_scores = semantic_model.score_candidates(histories, candidates)
            behavior_scores = behavior_model.score_candidates(
                histories,
                temporal_features,
                candidate_semantics,
                cand_ids_col,
                beh_dev_col,
            )
            sem_np = semantic_scores.detach().float().cpu().numpy()
            behavior_np = behavior_scores.detach().float().cpu().numpy()
            if not np.isfinite(sem_np).all() or not np.isfinite(behavior_np).all():
                sem_bad = ~np.isfinite(sem_np)
                beh_bad = ~np.isfinite(behavior_np)
                raise FloatingPointError(
                    "Non-finite SAGE or temporal-expert scores at dual-eval. "
                    f"batch={batch_index}, "
                    f"sem_nan={int(np.isnan(sem_np).sum())}, "
                    f"sem_inf={int(np.isinf(sem_np).sum())}, "
                    f"beh_nan={int(np.isnan(behavior_np).sum())}, "
                    f"beh_inf={int(np.isinf(behavior_np).sum())}"
                )

            semantic_chunks.append(sem_np)
            behavior_chunks.append(behavior_np)
            target_chunks.append(target_indices_np)
            set_progress_postfix(
                iterator,
                examples=sum(chunk.shape[0] for chunk in semantic_chunks),
                elapsed=format_seconds(clock.elapsed()),
            )
    finally:
        close_progress(iterator)

    if not semantic_chunks:
        raise RuntimeError("Prediction collection produced no batches.")
    return DualPredictionBundle(
        semantic_scores=np.concatenate(semantic_chunks, axis=0),
        behavior_scores=np.concatenate(behavior_chunks, axis=0),
        target_indices=np.concatenate(target_chunks, axis=0),
    )


def tune_static_beta(
    bundle: DualPredictionBundle,
    *,
    beta_grid: list[float],
    tie_policy: str,
    ks: tuple[int, ...],
    primary_metric: str,
    show_progress: bool | None = None,
    progress_desc: str = "static_fusion",
    progress_refresh: float = 0.5,
    leave_progress: bool = False,
) -> dict[str, Any]:
    """Grid-search static fusion beta on source validation."""
    semantic_rank, behavior_rank = bundle.ranks(tie_policy)
    semantic_z = zscore_rows(bundle.semantic_scores)
    behavior_z = zscore_rows(bundle.behavior_scores)

    values = [float(x) for x in beta_grid]
    trials: list[dict[str, Any]] = []
    clock = RunClock()
    bar = progress_bar(
        total=len(values),
        run_name=progress_desc,
        phase="TUNE",
        detail="static_beta",
        enabled=resolve_progress_enabled(show_progress),
        refresh_seconds=progress_refresh,
        leave=leave_progress,
    )
    try:
        for beta in values:
            fused_rank = ranks_from_scores(
                semantic_z + beta * behavior_z,
                bundle.target_indices,
                tie_policy,
            )
            mets = metrics_from_ranks(fused_rank, ks)
            trial = {"beta": beta, **mets}
            trials.append(trial)
            if bar is not None:
                bar.update(1)
            set_progress_postfix(
                bar,
                beta=beta,
                metric=mets[primary_metric],
                elapsed=format_seconds(clock.elapsed()),
            )
    finally:
        close_progress(bar)

    best = max(trials, key=lambda item: float(item[primary_metric]))
    return {
        "best_beta": float(best["beta"]),
        "best_metrics": best,
        "trials": trials,
        "semantic_metrics": metrics_from_ranks(semantic_rank, ks),
        "behavior_metrics": metrics_from_ranks(behavior_rank, ks),
        "oracle_metrics": metrics_from_ranks(
            np.minimum(semantic_rank, behavior_rank), ks
        ),
        "elapsed_seconds": float(clock.elapsed()),
    }
