from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def recall_at_k(ranks: np.ndarray | Sequence[float], k: int) -> float:
    """Recall@K for zero-based ranks."""
    ranks_np = np.asarray(ranks, dtype=np.float64)
    if ranks_np.size == 0:
        return 0.0
    return float((ranks_np < int(k)).mean())


def ndcg_at_k(ranks: np.ndarray | Sequence[float], k: int) -> float:
    """NDCG@K for one positive item and zero-based ranks."""
    ranks_np = np.asarray(ranks, dtype=np.float64)
    if ranks_np.size == 0:
        return 0.0
    hit = ranks_np < int(k)
    return float(np.where(hit, 1.0 / np.log2(ranks_np + 2.0), 0.0).mean())


def mrr_at_k(ranks: np.ndarray | Sequence[float], k: int) -> float:
    """MRR@K for one positive item and zero-based ranks."""
    ranks_np = np.asarray(ranks, dtype=np.float64)
    if ranks_np.size == 0:
        return 0.0
    hit = ranks_np < int(k)
    return float(np.where(hit, 1.0 / (ranks_np + 1.0), 0.0).mean())
