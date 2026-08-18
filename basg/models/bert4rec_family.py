from __future__ import annotations

import torch
from torch import nn


class BERT4RecSemanticFamily(nn.Module):
    """
    Shared BERT4Rec backbone for protocol-matched Sem, RecG, and SAGE baselines.

    architecture="single":
        BERT4Rec-Sem: recommendation projection only.

    architecture="dual":
        BERT4Rec-RecG/SAGE and the Arch0 diagnostic:
        recommendation projection + alignment projection + merge layer.

    Important project invariant:
    PrefixDataset is left padded, so the latest valid item is at index -1.
    """

    VALID_ARCHITECTURES = {"single", "dual"}

    def __init__(
        self,
        item_features: torch.Tensor,
        hidden_dim: int = 128,
        max_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        architecture: str = "single",
    ) -> None:
        super().__init__()
        if item_features.ndim != 2:
            raise ValueError(
                f"item_features must have shape [num_items+1, dim], got "
                f"{tuple(item_features.shape)}"
            )
        if architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {sorted(self.VALID_ARCHITECTURES)}, "
                f"got {architecture!r}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = int(hidden_dim)
        self.max_len = int(max_len)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.architecture = architecture
        self.embedding_dim = int(item_features.shape[1])

        # Excluded from state_dict so a source checkpoint can be loaded with a
        # target-domain feature table without resetting trained parameters.
        self.register_buffer(
            "item_features",
            item_features.detach().float(),
            persistent=False,
        )

        self.recommendation_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.alignment_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.merge_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.position_embedding = nn.Embedding(self.max_len, self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.recommendation_projection.weight)
        nn.init.zeros_(self.recommendation_projection.bias)
        nn.init.xavier_uniform_(self.alignment_projection.weight)
        nn.init.zeros_(self.alignment_projection.bias)
        nn.init.xavier_uniform_(self.merge_layer.weight)
        nn.init.zeros_(self.merge_layer.bias)

    @property
    def num_items(self) -> int:
        return int(self.item_features.shape[0] - 1)

    def set_item_features(self, item_features: torch.Tensor) -> None:
        """Switch catalog while preserving all trained model parameters."""
        if item_features.ndim != 2:
            raise ValueError("item_features must be two-dimensional")
        if int(item_features.shape[1]) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: checkpoint expects "
                f"{self.embedding_dim}, target has {int(item_features.shape[1])}"
            )
        device = next(self.parameters()).device
        self.item_features = item_features.detach().float().to(device)

    def raw_item_features(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_features[item_ids]

    def project_raw_for_alignment(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.alignment_projection(raw_features)

    def project_items_for_recommendation(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.recommendation_projection(self.raw_item_features(item_ids))

    def _sequence_inputs(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 2:
            raise ValueError(f"history must be [B, L], got {tuple(history.shape)}")
        if history.shape[1] != self.max_len:
            raise ValueError(
                f"Expected sequence length {self.max_len}, got {history.shape[1]}"
            )

        raw = self.raw_item_features(history)
        valid = history.ne(0).unsqueeze(-1)
        rec = self.recommendation_projection(raw)

        if self.architecture == "dual":
            align = self.alignment_projection(raw)
            x = self.merge_layer(torch.cat([rec, align], dim=-1))
        else:
            x = rec

        positions = torch.arange(
            self.max_len, device=history.device, dtype=torch.long
        ).unsqueeze(0)
        x = x + self.position_embedding(positions)
        x = self.input_dropout(x)
        # Prevent positional/bias leakage at PAD positions.
        return x * valid.to(x.dtype)

    def encode_sequence(self, history: torch.Tensor) -> torch.Tensor:
        x = self._sequence_inputs(history)
        key_padding_mask = history.eq(0)
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)
        encoded = encoded * history.ne(0).unsqueeze(-1).to(encoded.dtype)

        # PrefixDataset left-pads, so the most recent item is always at -1.
        return encoded[:, -1, :]

    def score_candidates(
        self,
        history: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_state = self.encode_sequence(history)
        candidate_state = self.project_items_for_recommendation(candidate_ids)
        return torch.einsum("bd,bcd->bc", user_state, candidate_state)

    def score_all_items(self, history: torch.Tensor) -> torch.Tensor:
        user_state = self.encode_sequence(history)
        all_ids = torch.arange(
            1, self.num_items + 1, device=history.device, dtype=torch.long
        )
        item_state = self.project_items_for_recommendation(all_ids)
        return user_state @ item_state.t()

    def export_hparams(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
        }
