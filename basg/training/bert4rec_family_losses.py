from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AlignmentLossOutput:
    total: torch.Tensor
    intra: torch.Tensor
    inter: torch.Tensor
    sic: torch.Tensor
    id_loss: torch.Tensor
    omega: torch.Tensor
    delta: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "align": float(self.total.detach().cpu()),
            "intra": float(self.intra.detach().cpu()),
            "inter": float(self.inter.detach().cpu()),
            "sic": float(self.sic.detach().cpu()),
            "id": float(self.id_loss.detach().cpu()),
            "omega": float(self.omega.detach().cpu()),
            "delta": float(self.delta.detach().cpu()),
        }


@dataclass
class RoleDistillationLossOutput:
    """Diagnostics for source next-item dynamic-role distillation."""

    total: torch.Tensor
    kl: torch.Tensor
    target_entropy: torch.Tensor
    prediction_entropy: torch.Tensor
    prediction_confidence: torch.Tensor
    valid_ratio: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "role": float(self.total.detach().cpu()),
            "role_kl": float(self.kl.detach().cpu()),
            "role_target_entropy": float(self.target_entropy.detach().cpu()),
            "role_pred_entropy": float(self.prediction_entropy.detach().cpu()),
            "role_pred_confidence": float(self.prediction_confidence.detach().cpu()),
            "role_valid_ratio": float(self.valid_ratio.detach().cpu()),
        }


def stable_bpr_loss(scores: torch.Tensor) -> torch.Tensor:
    """Numerically stable BPR for scores[:, 0]=positive, remaining=negatives."""
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must be [batch, 1 + num_negatives]")
    positive = scores[:, :1]
    negative = scores[:, 1:]
    return -F.logsigmoid(positive - negative).mean()


def dynamic_role_distillation_loss(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
    eps: float = 1e-8,
    normalize_targets: bool = True,
) -> RoleDistillationLossOutput:
    """
    KL(q_next_role || p_theta(next_role | prefix)) for source-only training.

    `target_distribution` may be a soft role distribution [B, K] or a non-normalized
    non-negative role score table looked up by target item id. Rows whose mass is
    <= eps are ignored so missing role observations do not inject noisy supervision.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [batch, num_roles], got {tuple(logits.shape)}")
    if target_distribution.ndim != 2:
        raise ValueError(
            "target_distribution must be [batch, num_roles], "
            f"got {tuple(target_distribution.shape)}"
        )
    if logits.shape != target_distribution.shape:
        raise ValueError(
            f"logits and target_distribution shape mismatch: "
            f"{tuple(logits.shape)} vs {tuple(target_distribution.shape)}"
        )

    target = target_distribution.to(device=logits.device, dtype=logits.dtype)
    target = torch.clamp(target, min=0.0)
    mass = target.sum(dim=-1, keepdim=True)
    valid = mass.squeeze(-1) > float(eps)

    zero = logits.sum() * 0.0
    if valid.sum() == 0:
        return RoleDistillationLossOutput(
            total=zero,
            kl=zero,
            target_entropy=zero,
            prediction_entropy=zero,
            prediction_confidence=zero,
            valid_ratio=torch.zeros((), dtype=logits.dtype, device=logits.device),
        )

    if normalize_targets:
        target = target / mass.clamp_min(float(eps))

    log_prob = F.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    per_example_kl = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)
    kl = per_example_kl[valid].mean()

    valid_target = target[valid]
    valid_prob = prob[valid]
    valid_log_prob = log_prob[valid]
    target_entropy = -(
        valid_target * valid_target.clamp_min(float(eps)).log()
    ).sum(dim=-1).mean()
    prediction_entropy = -(valid_prob * valid_log_prob).sum(dim=-1).mean()
    prediction_confidence = valid_prob.max(dim=-1).values.mean()
    valid_ratio = valid.to(logits.dtype).mean()

    return RoleDistillationLossOutput(
        total=kl,
        kl=kl,
        target_entropy=target_entropy,
        prediction_entropy=prediction_entropy,
        prediction_confidence=prediction_confidence,
        valid_ratio=valid_ratio,
    )


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    return -(prob * log_prob).sum(dim=-1).mean()


def _cosine_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), p=2, dim=-1, eps=1e-8)
    y = F.normalize(y.float(), p=2, dim=-1, eps=1e-8)
    return (x @ y.t()).clamp(-1.0, 1.0)


def recg_alignment_loss(
    projected: torch.Tensor,
    domain_labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> AlignmentLossOutput:
    """
    Official single-alpha RecG weighting.

    L_align = -alpha * H_intra
              + alpha * (N / |D|^3) * H_inter

    The returned value is already weighted and must be added exactly once.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    zero = _zero_like(projected)
    present = torch.unique(domain_labels, sorted=True)
    if present.numel() < 2:
        return AlignmentLossOutput(zero, zero, zero, zero, zero, zero, zero)

    intra_terms: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    center_domains: list[int] = []

    for domain in present.tolist():
        mask = domain_labels.eq(int(domain))
        values = projected[mask]
        if values.numel() == 0:
            continue
        centers.append(values.mean(dim=0))
        center_domains.append(int(domain))
        logits = _cosine_matrix(values, values) / float(temperature)
        intra_terms.append(_entropy_from_logits(logits))

    if len(centers) < 2:
        return AlignmentLossOutput(zero, zero, zero, zero, zero, zero, zero)

    intra = torch.stack(intra_terms).mean()
    centers_tensor = torch.stack(centers, dim=0)

    inter_terms: list[torch.Tensor] = []
    for domain in center_domains:
        values = projected[domain_labels.eq(domain)]
        other_index = [
            index for index, other_domain in enumerate(center_domains)
            if other_domain != domain
        ]
        if not other_index:
            continue
        other_centers = centers_tensor[other_index]
        logits = _cosine_matrix(values, other_centers) / float(temperature)
        inter_terms.append(_entropy_from_logits(logits))

    inter = torch.stack(inter_terms).mean() if inter_terms else zero
    num_samples = float(projected.shape[0])
    num_domains = float(len(center_domains))
    beta = float(alpha) * num_samples / (num_domains ** 3)
    total = -float(alpha) * intra + beta * inter
    return AlignmentLossOutput(total, intra, inter, zero, zero, zero, zero)


def sage_alignment_loss(
    projected: torch.Tensor,
    raw: torch.Tensor,
    domain_labels: torch.Tensor,
    source_domain_id: int,
    lambda_g: float,
    gamma_g: float,
    beta_id: float,
    temperature: float,
    adaptive_weight: bool = True,
) -> AlignmentLossOutput:
    """
    DPTF/SAGERec SAGE objective with one explicit outer weight.

    This preserves the public SAGERec definitions:
      - L_ID: mean entropy of source and pooled-auxiliary domains;
      - L_SIC: full source-by-auxiliary pairwise projected distance,
        weighted by positive raw-semantic cosine similarity;
      - delta: distance between raw-semantic domain centers;
      - omega = lambda_g * exp(-gamma_g * delta).

    The unified trainer adds the already weighted result exactly once:
        L_align = omega * (L_SIC + beta_id * L_ID)
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    zero = _zero_like(projected)
    source_mask = domain_labels.eq(int(source_domain_id))
    target_mask = ~source_mask
    if source_mask.sum() == 0 or target_mask.sum() == 0:
        return AlignmentLossOutput(zero, zero, zero, zero, zero, zero, zero)

    src_proj = projected[source_mask].float()
    tgt_proj = projected[target_mask].float()
    src_raw = raw[source_mask].float()
    tgt_raw = raw[target_mask].float()

    # DPTF L_ID averages the two binary super-domains.
    id_source = _entropy_from_logits(
        _cosine_matrix(src_proj, src_proj) / float(temperature)
    )
    id_target = _entropy_from_logits(
        _cosine_matrix(tgt_proj, tgt_proj) / float(temperature)
    )
    id_loss = (id_source + id_target) / 2.0

    # DPTF L_SIC uses every source-target pair, not index-wise pairing.
    projected_similarity = _cosine_matrix(src_proj, tgt_proj)
    projected_distance = (2.0 - 2.0 * projected_similarity).clamp_min(0.0)
    semantic_weight = _cosine_matrix(src_raw, tgt_raw).clamp_min(0.0)
    sic = (semantic_weight * projected_distance).sum() / (
        semantic_weight.sum() + 1e-8
    )

    # Domain-adaptive weight is based on raw semantic centers in SAGERec.
    src_center = F.normalize(
        src_raw.mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-8
    )
    tgt_center = F.normalize(
        tgt_raw.mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-8
    )
    delta = torch.linalg.vector_norm(src_center - tgt_center, ord=2)

    base_weight = torch.as_tensor(
        float(lambda_g), device=projected.device, dtype=projected.dtype
    )
    if adaptive_weight:
        omega = base_weight * torch.exp(-float(gamma_g) * delta.detach())
    else:
        omega = base_weight

    total = omega * (sic + float(beta_id) * id_loss)
    return AlignmentLossOutput(total, zero, zero, sic, id_loss, omega, delta)

def compute_alignment_loss(
    mode: str,
    projected: torch.Tensor,
    raw: torch.Tensor,
    domain_labels: torch.Tensor,
    config: dict,
) -> AlignmentLossOutput:
    mode = str(mode).lower()
    zero = _zero_like(projected)

    if mode in {"sem", "arch0"}:
        return AlignmentLossOutput(zero, zero, zero, zero, zero, zero, zero)

    temperature = float(config.get("temperature", 1.0))
    if mode == "recg":
        return recg_alignment_loss(
            projected=projected,
            domain_labels=domain_labels,
            alpha=float(config.get("alpha", 0.003)),
            temperature=temperature,
        )
    if mode == "sage":
        return sage_alignment_loss(
            projected=projected,
            raw=raw,
            domain_labels=domain_labels,
            source_domain_id=int(config.get("source_domain_id", 0)),
            lambda_g=float(config.get("lambda_g", 0.05)),
            gamma_g=float(config.get("gamma_g", 0.1)),
            beta_id=float(config.get("beta_id", 0.1)),
            temperature=temperature,
            adaptive_weight=bool(config.get("adaptive_weight", True)),
        )

    raise ValueError(f"Unknown baseline mode: {mode!r}")
