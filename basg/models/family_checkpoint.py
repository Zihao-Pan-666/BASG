from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from basg.models.bert4rec_family import BERT4RecSemanticFamily


def _torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def freeze_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze a loaded model for inference (eval mode, no gradients)."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_semantic_backbone(
    checkpoint_path: str | Path,
    item_features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load any supported semantic backbone from a checkpoint.

    Detects the architecture from ``model_hparams.architecture`` and
    instantiates the correct model class:

    - ``"single"`` / ``"dual"`` → BERT4RecSemanticFamily (SEM / RecG / SAGE)
    - ``"unisrec"``         → UniSRec (MoE + causal Transformer)

    All backbones expose ``score_candidates(histories, candidates)``,
    so the downstream fusion pipeline works without modification.
    """
    payload = _torch_load(checkpoint_path)
    hparams = payload["model_hparams"]
    arch = str(hparams.get("architecture", ""))

    if arch == "unisrec":
        from basg.baselines.unisrec import UniSRec

        model = UniSRec(
            item_features=item_features,
            hidden_dim=int(hparams["hidden_dim"]),
            max_len=int(hparams["max_len"]),
            num_layers=int(hparams["num_layers"]),
            num_heads=int(hparams["num_heads"]),
            dropout=float(hparams["dropout"]),
            n_exps=int(hparams.get("n_exps", 8)),
        )
    else:
        # BERT4Rec family: architecture ∈ {"single", "dual"}
        model = BERT4RecSemanticFamily(
            item_features=item_features,
            hidden_dim=int(hparams["hidden_dim"]),
            max_len=int(hparams["max_len"]),
            num_layers=int(hparams["num_layers"]),
            num_heads=int(hparams["num_heads"]),
            dropout=float(hparams["dropout"]),
            architecture=str(arch) if arch in {"single", "dual"} else "single",
        )

    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    freeze_for_inference(model)
    return model, payload


# Backward-compat alias (used by eval_family.py, train_family.py)
load_family_checkpoint = load_semantic_backbone
