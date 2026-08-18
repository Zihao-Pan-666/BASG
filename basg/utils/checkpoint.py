"""Checkpoint loading utilities — merged from duplicated script helpers.

Provides shared functions for loading BASG TemporalExpert and
BERT4Rec family checkpoints with consistent parameter extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from basg.models.temporal_expert import PopDynSequentialExpert


def load_temporal_expert(
    path: str | Path,
    device: torch.device,
) -> tuple[PopDynSequentialExpert, dict]:
    """Load a trained PopDynSequentialExpert checkpoint from disk.

    Args:
        path: path to the .pt checkpoint file.
        device: torch device to load the model onto.

    Returns:
        (model, checkpoint_payload) tuple. Model is set to eval mode
        with frozen parameters.
    """
    # Use weights_only=False because checkpoints may contain numpy arrays
    # (e.g., pca_components in behavior mapper payloads).
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
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


def load_temporal_expert_for_training(
    path: str | Path,
    device: torch.device,
) -> tuple[PopDynSequentialExpert, dict]:
    """Load a PopDynSequentialExpert checkpoint for continued training.

    Unlike load_temporal_expert, parameters remain trainable (no .eval(), no freeze).
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
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
    return model, payload


# Backward-compat alias (used by scripts 16, 17)
_load_temporal_expert = load_temporal_expert
