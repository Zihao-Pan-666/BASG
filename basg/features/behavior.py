"""AlphaFree-style behavior-augmented item representations.

Provides build_behavior_lookup() which runs the trained mapper over all
items to produce per-item behavior embeddings for candidate encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BehaviorMapper(nn.Module):
    """Map LLM embedding → behavior-augmented representation (PCA space)."""

    def __init__(self, semantic_dim: int, behavior_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, behavior_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def build_behavior_lookup(
    semantic_item_features: torch.Tensor,
    mapper_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Run behavior mapper over all items.

    Args:
        semantic_item_features: [num_items+1, semantic_dim] on CPU.
        mapper_path: path to saved behavior_mapper checkpoint.
        device: GPU device.

    Returns:
        [num_items+1, behavior_dim] float32 tensor on CPU.
    """
    ckpt = torch.load(mapper_path, map_location="cpu", weights_only=False)
    semantic_dim = semantic_item_features.shape[1]
    behavior_dim = ckpt["behavior_dim"]

    state = ckpt["mapper_state"]
    # Remap bare Sequential keys (0.weight) → wrapped keys (net.0.weight)
    if "net.0.weight" not in state and "0.weight" in state:
        state = {f"net.{k}": v for k, v in state.items()}
    model = BehaviorMapper(semantic_dim, behavior_dim)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    x = semantic_item_features.float().to(device)
    chunk_size = 4096
    results = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start: start + chunk_size]
        results.append(model(chunk).cpu())
    behavior = torch.cat(results, dim=0)  # [num_items+1, behavior_dim]
    return behavior
