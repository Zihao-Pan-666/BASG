"""Source-domain item popularity encoding (PrepRec-style).

Provides:
  1. build_pop_lookup() — run the popularity predictor over an item embedding
     table to produce a per-item popularity lookup [num_items+1].
  2. PopularityPredictor — small MLP trained to map LLM embedding → popularity
     percentile, enabling zero-shot compatible inference on target domains.
"""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path


class PopularityPredictor(nn.Module):
    """Map LLM semantic embedding → predicted popularity percentile [0, 1]."""

    def __init__(self, semantic_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw regression output — must match training (build_source_popularity.py
        # trains a bare Sequential without sigmoid).  Clamp to [0, 1] for safety.
        return self.net(x).clamp(0.0, 1.0)


@torch.no_grad()
def build_pop_lookup(
    semantic_item_features: torch.Tensor,
    predictor_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Run predictor over all items to build a popularity lookup table.

    Args:
        semantic_item_features: [num_items+1, semantic_dim] float tensor (CPU).
        predictor_path: path to saved PopularityPredictor state_dict.
        device: target device for inference.

    Returns:
        [num_items+1] float32 tensor on CPU, values in [0, 1].
    """
    ckpt = torch.load(predictor_path, map_location="cpu", weights_only=True)
    semantic_dim = semantic_item_features.shape[1]
    model = PopularityPredictor(semantic_dim)
    # Handle bare Sequential state_dict (keys like "0.weight") vs wrapped (keys like "net.0.weight")
    if isinstance(ckpt, dict) and "net.0.weight" not in ckpt and "0.weight" in ckpt:
        ckpt = {f"net.{k}": v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Process in chunks to avoid OOM for large catalogues
    x = semantic_item_features.float().to(device)
    chunk_size = 8192
    results: list[torch.Tensor] = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        pop = model(chunk)
        results.append(pop.cpu())
    pop = torch.cat(results, dim=0).squeeze(-1)  # [num_items+1]
    return pop
