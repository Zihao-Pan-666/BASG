"""UniSRec baseline for BASG -- causal Transformer with MoE adaptors.

Adapted from the SAGERec codebase for the BASG evaluation protocol.
Key simplifications vs the upstream version:
  - Sem mode only (no domain-alignment projection / merge layer)
  - BASG-native score_candidates(histories, candidates) interface
  - Compatible with evaluate_bert4rec_family() and PrefixDataset
  - Zero-shot via load_embeddings(new_item_features)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PWLayer — Parametric Whitening Layer (from UniSRec paper)
# ============================================================

class PWLayer(nn.Module):
    """Single Parametric Whitening Layer."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(self.dropout(x) - self.bias)


# ============================================================
# MoEAdaptorLayer — Mixture-of-Experts adaptor (UniSRec core)
# ============================================================

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced adaptor with noisy top-k gating."""

    def __init__(
        self,
        n_exps: int,
        layers: tuple[int, int],
        dropout: float = 0.0,
        noise: bool = True,
    ) -> None:
        super().__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise
        self.experts = nn.ModuleList(
            [PWLayer(layers[0], layers[1], dropout) for _ in range(n_exps)]
        )
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def _noisy_top_k_gating(
        self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2
    ) -> torch.Tensor:
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits).to(x.device) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits
        return F.softmax(logits, dim=-1)  # (..., n_exps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self._noisy_top_k_gating(x, self.training)  # (..., n_exps)
        expert_outputs = [
            self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)
        ]  # [(..., 1, D), ...]
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # (..., n_exps, D)
        return (gates.unsqueeze(-1) * expert_outputs).sum(dim=-2)  # (..., D)


# ============================================================
# UniSRec — causal Transformer + MoE projection for BASG
# ============================================================

class UniSRec(nn.Module):
    """UniSRec baseline for strict zero-shot cross-domain sequential recommendation.

    Architecture:
      item_features (frozen LLM embeddings)
        → MoEAdaptorLayer (8 experts, noisy gating)
        → causal Transformer encoder (SASRec-style)
        → dot-product scoring with candidate embeddings.

    Interface matches BASG evaluator expectations:
      - score_candidates(histories, candidates) → [B, C]
      - load_embeddings(new_features) for zero-shot domain switch
    """

    def __init__(
        self,
        item_features: torch.Tensor,
        hidden_dim: int = 128,
        max_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        n_exps: int = 8,
    ) -> None:
        super().__init__()
        if item_features.ndim != 2:
            raise ValueError(
                f"item_features must be [num_items+1, dim], got {tuple(item_features.shape)}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = int(hidden_dim)
        self.max_len = int(max_len)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.n_exps = int(n_exps)
        self.embedding_dim = int(item_features.shape[1])

        # Frozen item embedding table (padding idx 0)
        self.register_buffer(
            "item_features", item_features.detach().float(), persistent=False
        )

        # MoE projection: LLM embedding → hidden_dim
        self.projection_layer = MoEAdaptorLayer(
            n_exps=self.n_exps,
            layers=(self.embedding_dim, self.hidden_dim),
            dropout=self.dropout_rate,
        )

        self.pos_embedding = nn.Embedding(self.max_len, self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout_rate)

        # Causal Transformer encoder (SASRec-style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.ModuleList(
            [encoder_layer for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        # Lazy cache of MoE-projected item embeddings.
        # Invalidated on load_embeddings(); computed once on first forward pass.
        self.register_buffer(
            "_projected_cache", torch.empty(0), persistent=False
        )

    # ── Projection cache ──────────────────────────────────────────

    @torch.no_grad()
    def _ensure_projected(self) -> None:
        """Lazily project all item embeddings through the MoE adaptor.

        Called once at the start of training/evaluation.  The projected
        table is cached so subsequent score_candidates() calls are O(B*C),
        not O(N*D*E).
        """
        if (self._projected_cache.numel() == 0
                or self._projected_cache.shape[0] != self.item_features.shape[0]):
            was_training = self.training
            self.eval()
            self._projected_cache = self.projection_layer(self.item_features)
            if was_training:
                self.train()

    def _invalidate_cache(self) -> None:
        """Clear the projected-item cache (called after load_embeddings)."""
        self._projected_cache = torch.empty(0)

    # ── Sequence encoding ─────────────────────────────────────────

    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Encode a left-padded item sequence into a user state vector.

        Args:
            item_seq: [B, L] int64 tensor, 0 = padding.

        Returns:
            [B, hidden_dim] user representation from last valid position.
        """
        device = item_seq.device
        B, L = item_seq.shape
        valid_mask = (item_seq != 0).float().unsqueeze(-1)  # [B, L, 1]

        # Embed + project
        raw_emb = self.item_features[item_seq]               # [B, L, D]
        projected = self.projection_layer(raw_emb) * valid_mask  # [B, L, H]

        # Causal mask (SASRec style: cannot attend to future)
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )

        # Position encoding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.input_dropout(projected + self.pos_embedding(positions))

        # Transformer
        padding_mask = item_seq == 0
        for layer in self.encoder:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
            # NaN guard per layer
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.output_norm(x)
        x = x * valid_mask  # zero out padding positions

        # Extract last valid position (left-padded: last position is always valid
        # for non-empty sequences; for empty sequences mask ensures row is zero)
        user_state = x[:, -1, :]  # [B, H]
        return user_state

    # ── Candidate scoring (BASG evaluator interface) ─────────────

    def score_candidates(
        self,
        histories: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate items for each user history.

        Args:
            histories: [B, L] padded item IDs.
            candidates: [B, C] candidate item IDs.

        Returns:
            [B, C] dot-product scores.
        """
        user_state = self.encode_sequence(histories)         # [B, H]
        self._ensure_projected()
        candidate_emb = self._projected_cache[candidates]    # [B, C, H]
        scores = torch.einsum("bh,bch->bc", user_state, candidate_emb)
        return scores

    def score_from_state(
        self,
        user_state: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates from a pre-computed user state (training path).

        Avoids re-running the Transformer when scoring both positives
        and negatives against the same history.
        """
        self._ensure_projected()
        candidate_emb = self._projected_cache[candidate_ids]  # [B, C, H]
        return torch.einsum("bh,bch->bc", user_state, candidate_emb)

    # ── Zero-shot domain switching ────────────────────────────────

    @torch.no_grad()
    def load_embeddings(self, new_features: torch.Tensor) -> None:
        """Replace frozen item embedding table for zero-shot target inference."""
        if new_features.ndim != 2 or new_features.shape[1] != self.embedding_dim:
            raise ValueError(
                f"new_features must be [*, {self.embedding_dim}], "
                f"got {tuple(new_features.shape)}"
            )
        new_features = new_features.detach().float()
        if new_features[0].abs().sum() > 1e-6:
            new_features[0] = 0.0
        self.item_features.copy_(new_features)
        self._invalidate_cache()

    # ── Checkpoint compatibility ──────────────────────────────────

    def export_hparams(self) -> dict[str, int | float | str]:
        return {
            "architecture": "unisrec",
            "hidden_dim": self.hidden_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "n_exps": self.n_exps,
            "embedding_dim": self.embedding_dim,
        }
