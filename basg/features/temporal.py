"""Prefix temporal features — reduced to audit-proven complementary signals.

v7 (2026-07-08): Replaced 9 hand-crafted temporal features with 2 core temporal
features + 1 PrepRec-style source popularity percentile.  The original 9 features
(kept in v5/v6) showed diminishing returns when model capacity increased — the
bottleneck was input signal quality, not model size.

Kept:
  - gap_log_normalized  — per-user interaction interval (audit: correlated with
                           TemporalExpert complementarity on ACV)
  - recency_fraction    — how recently the item was interacted with

Added:
  - pop_percentile      — PrepRec-style item popularity percentile from source
                           domain, predicted via LLM→pop MLP for zero-shot compat.
"""

from __future__ import annotations

import torch


PREFIX_TEMPORAL_FEATURE_NAMES = (
    "gap_log_normalized",
    "recency_fraction",
    "pop_percentile",
)


def build_prefix_temporal_features(
    history_ids: torch.Tensor,
    history_times: torch.Tensor,
    *,
    time_scale_ms: float = 86_400_000.0,
    pop_lookup: torch.Tensor | None = None,
    use_temporal_features: bool = True,
    use_interaction_property: bool = True,
) -> torch.Tensor:
    """Build slim temporal + popularity tokens from ONE user's observed prefix.

    Catalog-free temporal features (gap_log_normalized, recency_fraction) are
    computed from per-user timestamps only.  When `pop_lookup` is provided
    (shape [num_items+1]), a per-item source popularity percentile is appended.
    Pop_lookup values are predicted from LLM embeddings via PopularityPredictor,
    enabling zero-shot compatible inference on target domains.
    """
    if history_ids.ndim != 2 or history_times.ndim != 2:
        raise ValueError("history_ids and history_times must both be [B, L].")
    if history_ids.shape != history_times.shape:
        raise ValueError("history_ids and history_times must have the same shape.")

    valid = history_ids.ne(0)
    batch_size, length = history_ids.shape
    if length == 0:
        raise ValueError("Prefix length cannot be zero.")

    times = history_times.to(dtype=torch.float64)
    times = torch.nan_to_num(times, nan=0.0, posinf=0.0, neginf=0.0)
    valid_f64 = valid.to(dtype=torch.float64)

    # ── gap_log_normalized ──────────────────────────────────────────
    previous_times = torch.cat(
        [torch.zeros((batch_size, 1), dtype=times.dtype, device=times.device), times[:, :-1]],
        dim=1,
    )
    previous_valid = torch.cat(
        [torch.zeros((batch_size, 1), dtype=torch.bool, device=valid.device), valid[:, :-1]],
        dim=1,
    )
    raw_gap = torch.where(valid & previous_valid, (times - previous_times).clamp_min(0.0), 0.0)
    scale = max(float(time_scale_ms), 1.0)
    log_gap = torch.log1p(raw_gap / scale)
    log_gap_max = torch.where(valid, log_gap, torch.zeros_like(log_gap)).amax(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    gap_log_normalized = log_gap / log_gap_max

    # ── recency_fraction ────────────────────────────────────────────
    inf = torch.full_like(times, float("inf"))
    first_time = torch.where(valid, times, inf).amin(dim=1, keepdim=True)
    first_time = torch.where(torch.isfinite(first_time), first_time, torch.zeros_like(first_time))
    last_time = torch.where(valid, times, torch.full_like(times, float("-inf"))).amax(
        dim=1, keepdim=True
    )
    last_time = torch.where(torch.isfinite(last_time), last_time, torch.zeros_like(last_time))
    span = (last_time - first_time).clamp_min(1.0)
    elapsed_fraction = ((times - first_time) / span).clamp(0.0, 1.0) * valid_f64
    recency_fraction = (1.0 - elapsed_fraction) * valid_f64

    # ── assemble base features ──────────────────────────────────────
    features_list = [
        gap_log_normalized.to(dtype=torch.float32).unsqueeze(-1),
        recency_fraction.to(dtype=torch.float32).unsqueeze(-1),
    ]

    # ── PrepRec-style source popularity percentile ──────────────────
    if pop_lookup is not None:
        if not isinstance(pop_lookup, torch.Tensor):
            pop_lookup = torch.as_tensor(pop_lookup, dtype=torch.float32)
        pop_lookup = pop_lookup.to(dtype=torch.float32, device=history_ids.device)
        pop_lookup = torch.nan_to_num(pop_lookup, nan=0.0, posinf=0.0, neginf=0.0)
        pop_lookup = pop_lookup.clamp(0.0, 1.0)
        if pop_lookup.shape[0] > 0:
            pop_lookup[0] = 0.0
        max_id = pop_lookup.shape[0] - 1
        clipped = history_ids.clamp(0, max_id)
        pop_val = pop_lookup[clipped].unsqueeze(-1)
        pop_val = pop_val * valid.unsqueeze(-1).to(dtype=torch.float32)
        pop_val = torch.nan_to_num(pop_val, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Always output 3 dims for consistent feature_dim
        pop_val = torch.zeros(batch_size, length, 1, dtype=torch.float32, device=history_ids.device)
    features_list.append(pop_val)

    features = torch.cat(features_list, dim=-1)  # all [B, L, 1] → [B, L, N]

    # ── Ablation: zero-fill temporal features (g, r at positions 0,1) ──
    if not use_temporal_features:
        features[:, :, 0] = 0.0  # gap_log_normalized
        features[:, :, 1] = 0.0  # recency_fraction

    # ── Ablation: zero-fill popularity (π̂ at position 2) ──
    if not use_interaction_property:
        features[:, :, 2] = 0.0  # pop_percentile

    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-20.0, 20.0)
    features = features.masked_fill((~valid).unsqueeze(-1), 0.0)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def last_valid_temporal_features(
    history_ids: torch.Tensor,
    temporal_features: torch.Tensor,
) -> torch.Tensor:
    """Return the latest valid token feature for each left-padded history."""
    if temporal_features.ndim != 3:
        raise ValueError("temporal_features must be [B, L, F].")
    valid_counts = history_ids.ne(0).sum(dim=1).clamp_min(1)
    indices = torch.full_like(valid_counts, temporal_features.shape[1] - 1) - (
        history_ids[:, -1].eq(0).long()
    )
    indices = indices.clamp(0, temporal_features.shape[1] - 1)
    rows = torch.arange(temporal_features.shape[0], device=temporal_features.device)
    return temporal_features[rows, indices]
