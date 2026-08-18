from __future__ import annotations

import math

import torch
from torch import nn


def _assert_finite(name: str, tensor: torch.Tensor, extra: dict | None = None) -> None:
    """Raise FloatingPointError with shape + first bad index + value range if tensor has NaN/Inf."""
    if not torch.isfinite(tensor).all():
        bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
        first_bad = bad[0].detach().cpu().tolist() if bad.numel() else None
        finite_vals = tensor[torch.isfinite(tensor)]
        info = {
            "shape": tuple(tensor.shape),
            "first_bad": first_bad,
            "finite_min": float(finite_vals.min().item()) if finite_vals.numel() else float("nan"),
            "finite_max": float(finite_vals.max().item()) if finite_vals.numel() else float("nan"),
            "finite_absmax": float(finite_vals.abs().max().item()) if finite_vals.numel() else float("nan"),
        }
        if extra:
            info.update(extra)
        raise FloatingPointError(
            f"{name} contains NaN/Inf; "
            + " ".join(f"{k}={v}" for k, v in info.items())
        )


class PopDynSequentialExpert(nn.Module):
    """Strict source-only temporal-behavior expert.

    v8 AlphaFree dual-branch: separate semantic and behavior projections
    on BOTH history and candidate sides, enabling sequence-behavior matching.
    """

    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        hidden_dim: int = 64,
        max_len: int = 50,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.15,
        position_mode: str = "fixed",
        behavior_dim: int = 0,
        behavior_gate_init: float = 0.1,
        fail_fast_behavior: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if position_mode not in {"fixed", "learned"}:
            raise ValueError("position_mode must be 'fixed' or 'learned'.")

        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_len = int(max_len)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.position_mode = str(position_mode)
        self.behavior_dim = int(behavior_dim)
        self.fail_fast_behavior = bool(fail_fast_behavior)

        # ── History / temporal side ──────────────────────────────────
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.sequence_norm = nn.LayerNorm(self.hidden_dim)

        # ── Candidate side: semantic branch (always present) ─────────
        self.semantic_norm = nn.LayerNorm(self.semantic_dim)
        self.semantic_proj = nn.Linear(self.semantic_dim, self.hidden_dim)

        # ── Candidate side: behavior branch (v8 dual-branch) ─────────
        if self.behavior_dim > 0:
            self.behavior_norm = nn.LayerNorm(self.behavior_dim)
            self.behavior_proj = nn.Linear(self.behavior_dim, self.hidden_dim)
            self.behavior_gate = nn.Parameter(torch.tensor(float(behavior_gate_init)))
            # History-side behavior projection
            self.history_behavior_proj = nn.Linear(self.behavior_dim, self.hidden_dim)
            self.history_behavior_gate = nn.Parameter(torch.tensor(float(behavior_gate_init)))
        else:
            self.behavior_norm = None
            self.behavior_proj = None
            self.behavior_gate = None
            self.history_behavior_proj = None
            self.history_behavior_gate = None

        # ── Shared candidate encoder (hidden → hidden) ───────────────
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.candidate_output_norm = nn.LayerNorm(self.hidden_dim)

        # ── Transformer ──────────────────────────────────────────────
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.position_mode == "fixed":
            self.register_buffer(
                "position_encoding",
                self._sinusoidal_encoding(self.max_len, self.hidden_dim),
                persistent=False,
            )
            self.learned_position = None
        else:
            self.position_encoding = None
            self.learned_position = nn.Embedding(self.max_len, self.hidden_dim)

    @staticmethod
    def _sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(dim, 1))
        )
        out = torch.zeros(length, dim, dtype=torch.float32)
        out[:, 0::2] = torch.sin(position * scale)
        out[:, 1::2] = torch.cos(position * scale)
        return out

    def _position(self, length: int, device: torch.device) -> torch.Tensor:
        if self.position_mode == "fixed":
            assert self.position_encoding is not None
            return self.position_encoding[:length].to(device=device).unsqueeze(0)
        assert self.learned_position is not None
        positions = torch.arange(length, dtype=torch.long, device=device)
        return self.learned_position(positions).unsqueeze(0)

    # ═════════════════════════════════════════════════════════════════
    # History / sequence encoding
    # ═════════════════════════════════════════════════════════════════

    def encode_sequence(
        self,
        history_ids: torch.Tensor,
        history_features: torch.Tensor,
        behavior_lookup: torch.Tensor | None = None,
        use_interaction_property: bool = True,
    ) -> torch.Tensor:
        if history_features.ndim != 3:
            raise ValueError("history_features must have shape [B, L, F].")
        if history_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: {history_features.shape[-1]} vs {self.feature_dim}"
            )

        _assert_finite("history_features_before_input_norm", history_features)

        valid_mask = history_ids.ne(0)
        has_valid = valid_mask.any(dim=1)
        B, L = history_ids.shape

        # ── Safe padding mask ────────────────────────────────────────
        safe_valid_mask = valid_mask.clone()
        all_pad_rows = ~has_valid
        if all_pad_rows.any():
            safe_valid_mask[all_pad_rows, -1] = True
        safe_key_padding_mask = ~safe_valid_mask

        # ── Temporal feature encoding ────────────────────────────────
        normed = self.input_norm(history_features)
        _assert_finite("after_input_norm", normed)
        x = self.feature_encoder(normed)
        _assert_finite("after_feature_encoder", x)

        # ── v8: history-side behavior fusion ─────────────────────────
        if use_interaction_property and self.behavior_dim > 0 and behavior_lookup is not None:
            clipped = history_ids.clamp(0, behavior_lookup.shape[0] - 1)
            h_beh = behavior_lookup[clipped].to(dtype=x.dtype, device=x.device)
            # Padding item id=0 → zero behavior
            h_beh = h_beh * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
            _assert_finite("history_behavior_features", h_beh)
            h_beh_proj = self.history_behavior_proj(h_beh)
            x = x + self.history_behavior_gate * h_beh_proj
        elif self.behavior_dim > 0 and behavior_lookup is None and self.fail_fast_behavior:
            raise RuntimeError(
                "behavior_dim > 0 but behavior_lookup is None in encode_sequence. "
                "Set fail_fast_behavior=False to allow silent fallback."
            )

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp_(-10.0, 10.0)
        x = x + self._position(L, history_ids.device).to(dtype=x.dtype)
        x = self.dropout(x)
        x = x.masked_fill((~valid_mask).unsqueeze(-1), 0.0)

        _assert_finite("before_transformer_encoder", x)

        # ── Transformer ──────────────────────────────────────────────
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=history_ids.device), diagonal=1,
        )
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=safe_key_padding_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        if has_valid.any():
            _assert_finite("after_transformer_encoder_valid_rows", encoded[has_valid])
        encoded = torch.where(has_valid.view(-1, 1, 1), encoded, torch.zeros_like(encoded))

        # ── Last valid token ─────────────────────────────────────────
        positions = torch.arange(L, device=history_ids.device).unsqueeze(0)
        last_valid_idx = torch.where(
            valid_mask, positions.expand_as(history_ids), torch.zeros_like(history_ids),
        ).max(dim=1).values
        last_valid_idx = last_valid_idx.clamp(0, L - 1)
        rows = torch.arange(B, device=history_ids.device)
        last_hidden = encoded[rows, last_valid_idx]

        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        last_hidden = last_hidden.clamp_(-10.0, 10.0)
        out = self.sequence_norm(last_hidden)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.clamp_(-10.0, 10.0)
        out = torch.where(has_valid.unsqueeze(-1), out, torch.zeros_like(out))
        _assert_finite("sequence_state", out)
        return out

    # ═════════════════════════════════════════════════════════════════
    # Candidate encoding (v8 dual-branch)
    # ═════════════════════════════════════════════════════════════════

    def encode_candidates(
        self,
        candidate_semantic_features: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
        behavior_lookup: torch.Tensor | None = None,
        use_interaction_property: bool = True,
    ) -> torch.Tensor:
        if candidate_semantic_features.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Candidate semantic dim mismatch: "
                f"{candidate_semantic_features.shape[-1]} vs {self.semantic_dim}"
            )
        _assert_finite("candidate_semantic_features", candidate_semantic_features)

        # Semantic branch
        sem_hidden = self.semantic_proj(self.semantic_norm(candidate_semantic_features))

        # Behavior branch (v8 dual-branch) — explicit boolean gate
        if use_interaction_property and self.behavior_dim > 0 and behavior_lookup is not None and candidate_item_ids is not None:
            clipped = candidate_item_ids.clamp(0, behavior_lookup.shape[0] - 1)
            beh = behavior_lookup[clipped].to(
                dtype=candidate_semantic_features.dtype,
                device=candidate_semantic_features.device,
            )
            _assert_finite("candidate_behavior_features", beh)
            beh_hidden = self.behavior_proj(self.behavior_norm(beh))
            x = sem_hidden + self.behavior_gate * beh_hidden
        elif use_interaction_property and self.behavior_dim > 0 and self.fail_fast_behavior:
            raise RuntimeError(
                "behavior_dim > 0 but behavior_lookup or candidate_item_ids is None "
                "in encode_candidates. Set fail_fast_behavior=False to allow silent fallback."
            )
        else:
            x = sem_hidden

        x = self.candidate_encoder(x)
        x = x.clamp_(-50.0, 50.0)
        return self.candidate_output_norm(x)

    # ═════════════════════════════════════════════════════════════════
    # Scoring
    # ═════════════════════════════════════════════════════════════════

    def _score_encoded(
        self,
        state: torch.Tensor,
        candidate_state: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        candidate_state = torch.nan_to_num(candidate_state, nan=0.0, posinf=0.0, neginf=0.0)
        scores = torch.einsum("bd,bcd->bc", state, candidate_state)
        _assert_finite("temporal_expert_scores", scores)
        return scores

    def score_candidates(
        self,
        history_ids: torch.Tensor,
        history_features: torch.Tensor,
        candidate_semantic_features: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
        behavior_lookup: torch.Tensor | None = None,
        use_interaction_property: bool = True,
    ) -> torch.Tensor:
        # v8: behavior_lookup flows to BOTH encode_sequence and encode_candidates
        state = self.encode_sequence(history_ids, history_features, behavior_lookup,
                                     use_interaction_property=use_interaction_property)
        candidate_state = self.encode_candidates(
            candidate_semantic_features, candidate_item_ids, behavior_lookup,
            use_interaction_property=use_interaction_property,
        )
        return self._score_encoded(state, candidate_state)

    def export_hparams(self) -> dict[str, int | float | str]:
        h = {
            "feature_dim": self.feature_dim,
            "semantic_dim": self.semantic_dim,
            "hidden_dim": self.hidden_dim,
            "max_len": self.max_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "position_mode": self.position_mode,
            "behavior_dim": self.behavior_dim,
            "behavior_gate": float(self.behavior_gate.item()) if self.behavior_gate is not None else 0.0,
            "history_behavior_gate": float(self.history_behavior_gate.item()) if self.history_behavior_gate is not None else 0.0,
        }
        return h

    def behavior_diagnostics(self) -> dict[str, float]:
        """Return gate values and weight norms for logging."""
        d = {"behavior_dim": self.behavior_dim}
        if self.behavior_gate is not None:
            d["behavior_gate"] = float(self.behavior_gate.item())
            d["behavior_proj_wn"] = float(self.behavior_proj.weight.norm().item())
        if self.history_behavior_gate is not None:
            d["history_behavior_gate"] = float(self.history_behavior_gate.item())
            d["history_behavior_proj_wn"] = float(self.history_behavior_proj.weight.norm().item())
        if hasattr(self, "semantic_proj"):
            d["semantic_proj_wn"] = float(self.semantic_proj.weight.norm().item())
        return d
