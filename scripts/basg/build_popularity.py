#!/usr/bin/env python
"""Build source-domain item popularity percentile table (PrepRec-style).

Computes per-item per-week popularity percentiles using exponential
weighted averaging over interaction counts.  Only source-domain data
is touched — no target leakage.

Output:
  artifacts/popularity/{source}_pop_table.pt  — [num_weeks, num_items+1] float32
  artifacts/popularity/{source}_pop_predictor.pt — small MLP (LLM_embed → percentile)

Usage:
  python scripts/build_source_popularity.py --source amazon_movies_and_tv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from basg.features.semantic import load_semantic_embeddings
from basg.utils.bert4rec_family_compat import load_interactions_compat
from basg.utils.runtime import log_event


def _week_index(ts_ms: np.ndarray, origin_ms: float) -> np.ndarray:
    """Convert millisecond Unix timestamps to zero-based week indices."""
    return ((ts_ms - origin_ms) / (7 * 86400 * 1000)).astype(np.int64)


def build_pop_table(source: str, data_root: str = "./data") -> tuple[torch.Tensor, dict]:
    """Build exponential-weighted popularity percentile table for source domain."""
    interactions, _, item_map = load_interactions_compat(data_root, source)
    num_items = len(item_map)

    # Sort by timestamp
    interactions = interactions.sort_values("Timestamp")
    # The processed CSV already has integer ItemId (1..num_items) from item_map.
    # No mapping needed — use directly.
    timestamps = interactions["Timestamp"].values.astype(np.int64)
    item_ids = interactions["ItemId"].values.astype(np.int64)

    origin_ms = float(timestamps.min())
    weeks = _week_index(timestamps, origin_ms)
    num_weeks = int(weeks.max()) + 1

    log_event("POP", "BUILD_START", source=source, items=num_items, weeks=num_weeks)

    # Exponential-weighted interaction counts per week
    # weight=0.5 as in PrepRec data_full.py
    weight = 0.5
    counter = np.zeros(num_items + 1, dtype=np.float64)
    pop_raw = np.zeros((num_weeks, num_items + 1), dtype=np.float32)

    for w in range(num_weeks):
        mask = weeks == w
        if mask.any():
            w_items = item_ids[mask]
            unique, counts = np.unique(w_items, return_counts=True)
            counter[unique] += counts.astype(np.float64)
        # Decay
        counter *= weight
        # Percentile rank (0-100), zero for unseen items
        nonzero = counter > 0
        if nonzero.sum() > 1:
            ranks = rankdata(counter[nonzero], method="average")
            percs = 100.0 * ranks / (ranks.max() + 1)
            pop_raw[w, nonzero] = percs.astype(np.float32)
        if w % 200 == 0:
            log_event("POP", "PROGRESS", week=w, total=num_weeks)

    log_event("POP", "BUILD_DONE", shape=tuple(pop_raw.shape))
    return torch.from_numpy(pop_raw), {
        "num_weeks": num_weeks,
        "num_items": num_items,
        "origin_ms": origin_ms,
        "weight": weight,
    }


def train_pop_predictor(
    pop_table: torch.Tensor,
    source: str,
    data_root: str = "./data",
    epochs: int = 20,
    lr: float = 1e-3,
) -> torch.nn.Module:
    """Train a small MLP that predicts popularity percentile from LLM embedding.

    This enables zero-shot compatible popularity estimates for target-domain
    items at inference time: LLM_embed(target_item) → predicted_popularity.
    """
    _, _, item_map = load_interactions_compat(data_root, source)
    item_features = load_semantic_embeddings(
        data_root=data_root, domain=source, item_map=item_map
    )  # [num_items+1, 4096]

    semantic_dim = item_features.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_features = item_features.float().to(device)

    # Training target: mean popularity percentile over the last 8 weeks
    # (stable estimate of "overall popularity level")
    recent_pop = pop_table[-min(8, pop_table.shape[0]):].mean(dim=0).to(device)
    target = recent_pop / 100.0  # normalize to [0, 1]

    # Only train on items that have some popularity signal
    valid = target > 0.001
    log_event("POP", "PREDICTOR_TRAIN", valid_items=int(valid.sum().item()), total_items=len(valid))

    model = torch.nn.Sequential(
        torch.nn.Linear(semantic_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    x = item_features[valid]
    y = target[valid].unsqueeze(-1)

    n = len(x)
    batch_size = 4096
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)
        if epoch % 5 == 0:
            log_event("POP", "PREDICTOR_EPOCH", epoch=epoch, loss=total_loss / n)

    model.eval()
    return model.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source popularity table")
    parser.add_argument("--source", required=True, help="Source domain name")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--no-predictor", action="store_true", help="Skip predictor training")
    args = parser.parse_args()

    out_dir = Path("artifacts/popularity")
    out_dir.mkdir(parents=True, exist_ok=True)

    pop_table, meta = build_pop_table(args.source, args.data_root)
    table_path = out_dir / f"{args.source}_pop_table.pt"
    torch.save({"pop_table": pop_table, "meta": meta}, table_path)
    log_event("ARTIFACT", "SAVED", path=str(table_path))

    if not args.no_predictor:
        predictor = train_pop_predictor(pop_table, args.source, args.data_root)
        pred_path = out_dir / f"{args.source}_pop_predictor.pt"
        torch.save(predictor.state_dict(), pred_path)
        log_event("ARTIFACT", "SAVED", path=str(pred_path))

    log_event("POP", "DONE")


if __name__ == "__main__":
    main()
