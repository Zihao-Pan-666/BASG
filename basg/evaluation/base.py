"""Shared evaluation utilities used across all evaluators.

Extracted from bert4rec_family_evaluator and dual_expert_evaluator
to eliminate code duplication.
"""

from __future__ import annotations

import numpy as np


# ── Negative sampling ────────────────────────────────────────────────

def sample_negatives(
    num_items: int,
    blocked: set[int],
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample negative item IDs avoiding blocked items (history + target)."""
    available = num_items - len([x for x in blocked if 1 <= x <= num_items])
    if available <= 0:
        raise ValueError("No candidate negatives remain after blocking history/target")

    replace = available < count
    if replace:
        pool = np.asarray(
            [item for item in range(1, num_items + 1) if item not in blocked],
            dtype=np.int64,
        )
        return rng.choice(pool, size=count, replace=True).astype(int).tolist()

    result: set[int] = set()
    while len(result) < count:
        draw = rng.integers(1, num_items + 1, size=max(32, count * 2))
        for value in draw.tolist():
            item = int(value)
            if item not in blocked:
                result.add(item)
                if len(result) == count:
                    break
    return list(result)


# ── Ranking with ties ─────────────────────────────────────────────────

def rank_with_ties(scores: np.ndarray, target_index: int, policy: str) -> int:
    """Compute 0-based rank of target_index in scores, handling ties."""
    target = float(scores[target_index])
    greater = int(np.sum(scores > target))
    equal_others = int(np.sum(scores == target)) - 1

    if policy == "worst":
        return greater + max(equal_others, 0)
    if policy == "best":
        return greater
    if policy == "average":
        return int(round(greater + max(equal_others, 0) / 2.0))
    raise ValueError(f"Unknown tie policy: {policy}")


def ranks_from_scores(
    scores: np.ndarray,
    target_indices: np.ndarray,
    tie_policy: str,
) -> np.ndarray:
    """Compute ranks for a batch of score rows."""
    return np.asarray(
        [
            rank_with_ties(row, int(target_index), tie_policy)
            for row, target_index in zip(scores, target_indices)
        ],
        dtype=np.int64,
    )


# ── Metrics computation ───────────────────────────────────────────────

def metrics_from_ranks(ranks: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    """Compute Recall, NDCG, MRR from 0-based ranks."""
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.size == 0:
        raise RuntimeError("No ranks were collected.")
    metrics: dict[str, float] = {"num_examples": float(ranks.size)}
    for k in sorted(set(int(x) for x in ks)):
        hit = ranks < k
        metrics[f"Recall@{k}"] = float(hit.mean())
        metrics[f"NDCG@{k}"] = float(
            np.where(hit, 1.0 / np.log2(ranks + 2.0), 0.0).mean()
        )
        metrics[f"MRR@{k}"] = float(
            np.where(hit, 1.0 / (ranks + 1.0), 0.0).mean()
        )
    metrics["mean_rank_1based"] = float((ranks + 1).mean())
    return metrics


# ── Candidate batch construction ──────────────────────────────────────

def build_candidate_batch(
    histories: list[list[int]],
    targets: list[int],
    *,
    num_items: int,
    eval_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sampled candidate sets + record target positions."""
    candidates: list[list[int]] = []
    target_indices: list[int] = []
    for history, target in zip(histories, targets):
        blocked = {int(item) for item in history if int(item) > 0}
        blocked.add(int(target))
        negatives = sample_negatives(
            num_items=num_items,
            blocked=blocked,
            count=int(eval_negatives),
            rng=rng,
        )
        row = negatives + [int(target)]
        rng.shuffle(row)
        candidates.append(row)
        target_indices.append(row.index(int(target)))
    return np.asarray(candidates, dtype=np.int64), np.asarray(target_indices, dtype=np.int64)


# ── Score diagnostics ─────────────────────────────────────────────────

def score_diagnostics(scores: np.ndarray) -> dict[str, float]:
    """Compute tie/equality diagnostics for a score matrix [N, C]."""
    if scores.ndim != 2:
        raise ValueError("scores must be [N, C].")
    all_equal = np.isclose(np.ptp(scores, axis=1), 0.0)
    tie_rows = np.zeros(scores.shape[0], dtype=bool)
    largest_tie = np.ones(scores.shape[0], dtype=np.float32)
    for index, row in enumerate(scores):
        _, counts = np.unique(np.round(row, decimals=12), return_counts=True)
        largest_tie[index] = float(counts.max())
        tie_rows[index] = bool((counts > 1).any())
    return {
        "all_equal_ratio": float(all_equal.mean()),
        "tie_case_ratio": float(tie_rows.mean()),
        "avg_tie_items": float(largest_tie.mean()),
    }


# ── Score normalization ───────────────────────────────────────────────

def zscore_rows(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Row-wise z-score normalization."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.maximum(std, eps)


def entropy_and_margin(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute row-wise entropy (0-1) and top1-top2 margin from scores."""
    standardized = zscore_rows(scores)
    standardized = standardized - standardized.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(standardized, -30.0, 30.0))
    probability = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
    entropy = entropy / np.log(max(scores.shape[1], 2))
    if scores.shape[1] < 2:
        margin = np.zeros(scores.shape[0], dtype=np.float32)
    else:
        top2 = np.partition(standardized, -2, axis=1)[:, -2:]
        margin = top2.max(axis=1) - top2.min(axis=1)
    return entropy.astype(np.float32), margin.astype(np.float32)


# ── Deprecated compatibility aliases (private names from old code) ────

_sample_negatives = sample_negatives
_rank_with_ties = rank_with_ties
_ranks_from_scores = ranks_from_scores
_metrics_from_ranks = metrics_from_ranks
_candidate_batch = build_candidate_batch
_score_diagnostics = score_diagnostics
_zscore_rows = zscore_rows
_entropy_and_margin = entropy_and_margin
