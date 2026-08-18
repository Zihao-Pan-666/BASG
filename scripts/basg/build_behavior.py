#!/usr/bin/env python
"""Build AlphaFree-style behavior-augmented item representations from source.

Phase 0 — diagnostics only, no TemporalExpert training.

Steps:
  1. Build item-user sparse matrix A from AMT interactions.
  2. Compute item-item co-occurrence S = A^T @ A (3 modes: log_count/ppmi/cosine).
  3. For each item i, select top-K=100 co-occurrence neighbors.
  4. Apply LLM cosine soft semantic filter.
  5. z_i^+ = weighted average of neighbor LLM embeddings (4096-dim).
  6. PCA on all items → 64 dim.
  7. Item-level holdout split (stratified by degree: head/mid/tail).
  8. Train mapper on train split only: LLM_i → z_i^+_64.
  9. Evaluate mapper on train, heldout, head, mid, tail.
  10. Save JSON audit report + artifacts.

Usage:
  python scripts/build_source_behavior_embedding.py --source amazon_movies_and_tv
  python scripts/build_source_behavior_embedding.py --source amazon_movies_and_tv --holdout-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from basg.features.semantic import load_semantic_embeddings
from basg.utils.bert4rec_family_compat import load_interactions_compat
from basg.utils.runtime import log_event


# ═══════════════════════════════════════════════════════════════════════════════
# Co-occurrence builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cooc_log_count(A, n_items):
    S_raw = A.T.dot(A)
    S = S_raw.copy()
    S.data = np.log1p(S.data)
    deg = np.array(S.sum(axis=1)).ravel()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg, dtype=np.float64))
    D_inv = sparse.diags(deg_inv_sqrt)
    return D_inv.dot(S).dot(D_inv), deg


def _build_cooc_ppmi(A, n_items, shift=1.0):
    S_raw = A.T.dot(A)
    total = S_raw.sum()
    deg = np.array(S_raw.sum(axis=1)).ravel()
    expected = np.maximum(np.outer(deg, deg) / total, 1e-10)
    S = S_raw.copy()
    S.data = np.maximum(0.0, np.log(np.maximum(S.data, 1e-10) / expected[S.row, S.col]) - shift)
    S.eliminate_zeros()
    row_sum = np.array(S.sum(axis=1)).ravel()
    row_sum_inv = np.power(row_sum, -1.0, where=row_sum > 0, out=np.zeros_like(row_sum, dtype=np.float64))
    return sparse.diags(row_sum_inv).dot(S), deg


def _build_cooc_cosine(A, n_items):
    S_raw = A.T.dot(A)
    deg = np.array(S_raw.sum(axis=1)).ravel()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg, dtype=np.float64))
    return sparse.diags(deg_inv_sqrt).dot(S_raw).dot(sparse.diags(deg_inv_sqrt)), deg


BUILDERS = {"log_count": _build_cooc_log_count, "ppmi": _build_cooc_ppmi, "cosine": _build_cooc_cosine}


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic filter
# ═══════════════════════════════════════════════════════════════════════════════

def _soft_semantic_filter(neighbors, weights, emb_i, all_embs, mu_i, tau=0.1):
    n_ids = np.array(neighbors, dtype=np.int64)
    sims = np.dot(all_embs[n_ids], emb_i)
    norms_i = np.linalg.norm(emb_i)
    norms_j = np.linalg.norm(all_embs[n_ids], axis=1)
    cos_sims = sims / np.maximum(norms_i * norms_j, 1e-8)
    gate = 1.0 / (1.0 + np.exp(-(cos_sims - mu_i) / max(tau, 1e-4)))
    return weights * gate


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior representation builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_behavior_representations(source, data_root="./data", K_c=100, tau=0.1, cooc_mode="log_count"):
    interactions, _, item_map = load_interactions_compat(data_root, source)
    n_items = len(item_map)
    n_users = interactions["UserId"].nunique()
    log_event("BEHAVIOR", "BUILD_START", source=source, items=n_items, users=n_users, K_c=K_c, mode=cooc_mode)

    user_ids = interactions["UserId"].astype("category").cat.codes.values.astype(np.int64)
    item_ids = interactions["ItemId"].values.astype(np.int64)
    vals = np.ones(len(item_ids), dtype=np.float32)
    A = sparse.csr_matrix((vals, (user_ids, item_ids)), shape=(n_users, n_items + 1))

    S_norm, deg = BUILDERS[cooc_mode](A, n_items + 1)
    log_event("BEHAVIOR", "COOC_DONE", mode=cooc_mode, nnz=int(S_norm.nnz),
              deg_mean=float(deg.mean()), deg_median=float(np.median(deg)))

    # Load LLM embeddings
    item_features = load_semantic_embeddings(data_root=data_root, domain=source, item_map=item_map)
    item_features = item_features.float().numpy()
    emb_norm = normalize(item_features, norm="l2", axis=1)
    n_total = emb_norm.shape[0]
    mu_per_item = np.zeros(n_total, dtype=np.float32)
    batch = 2048
    for start in range(0, n_total, batch):
        end = min(start + batch, n_total)
        block = emb_norm[start:end] @ emb_norm.T
        mu_per_item[start:end] = block.mean(axis=1)

    # Build z_plus
    z_plus = np.zeros_like(item_features, dtype=np.float32)
    neighbor_stats = {"degrees": [], "cos_sims": [], "head_ratio": []}
    head_threshold = np.percentile(deg[1:], 90)

    for i in range(1, n_items + 1):
        row = S_norm.getrow(i)
        if row.nnz == 0:
            continue
        indices = row.indices
        data = row.data.copy()
        mask = indices != i
        indices, data = indices[mask], data[mask]
        if len(indices) == 0:
            continue
        topk = min(K_c, len(indices))
        topk_idx = np.argpartition(data, -topk)[-topk:]
        topk_data = data[topk_idx]
        topk_indices = indices[topk_idx]
        topk_data = _soft_semantic_filter(topk_indices, topk_data, emb_norm[i], emb_norm, mu_per_item[i], tau=tau)
        w_sum = topk_data.sum()
        if w_sum > 0:
            topk_data = topk_data / w_sum
            z_plus[i] = (item_features[topk_indices] * topk_data[:, None]).sum(axis=0)
            neighbor_stats["degrees"].append(deg[topk_indices].mean())
            neighbor_stats["cos_sims"].append((emb_norm[topk_indices] @ emb_norm[i]).mean())
            neighbor_stats["head_ratio"].append((deg[topk_indices] > head_threshold).mean())
        if i % 20000 == 0:
            log_event("BEHAVIOR", "PROGRESS", item=i, total=n_items)
    z_plus[0] = 0.0

    deg_arr = np.array(neighbor_stats["degrees"])
    cos_arr = np.array(neighbor_stats["cos_sims"])
    head_arr = np.array(neighbor_stats["head_ratio"])
    log_event("BEHAVIOR", "DIAG_NEIGHBOR",
              deg_mean=float(deg_arr.mean()), deg_median=float(np.median(deg_arr)),
              deg_p10=float(np.percentile(deg_arr, 10)), deg_p90=float(np.percentile(deg_arr, 90)),
              cos_mean=float(cos_arr.mean()), cos_median=float(np.median(cos_arr)),
              cos_p10=float(np.percentile(cos_arr, 10)), cos_p90=float(np.percentile(cos_arr, 90)),
              head_ratio=float(head_arr.mean()),
              items_with_neighbors=int((z_plus.sum(axis=1) != 0).sum()))

    return z_plus, item_features, deg


# ═══════════════════════════════════════════════════════════════════════════════
# Mapper training with item-level holdout
# ═══════════════════════════════════════════════════════════════════════════════

def _covariance_effective_rank(X, threshold=0.95):
    """Effective rank: number of PCs needed to explain `threshold` variance."""
    _, s, _ = np.linalg.svd(X.astype(np.float64) - X.mean(axis=0), full_matrices=False)
    s2 = s ** 2
    cumsum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cumsum, threshold) + 1)


def _stratify_by_degree(deg, valid_idx, holdout_ratio=0.2, seed=42):
    """Split items into train/holdout, stratified by degree tertiles."""
    rng = np.random.default_rng(seed)
    deg_valid = deg[valid_idx]
    p33, p67 = np.percentile(deg_valid, [33, 67])
    strata = np.zeros(len(valid_idx), dtype=int)
    strata[deg_valid <= p33] = 0       # tail
    strata[(deg_valid > p33) & (deg_valid <= p67)] = 1  # mid
    strata[deg_valid > p67] = 2        # head

    train_idx, heldout_idx = [], []
    for s in range(3):
        s_idx = valid_idx[strata == s]
        rng.shuffle(s_idx)
        n_hold = max(1, int(len(s_idx) * holdout_ratio))
        heldout_idx.append(s_idx[:n_hold])
        train_idx.append(s_idx[n_hold:])
    train_idx = np.concatenate(train_idx)
    heldout_idx = np.concatenate(heldout_idx)
    rng.shuffle(train_idx)
    rng.shuffle(heldout_idx)
    return train_idx, heldout_idx, strata


def _cosine_safe(a, b):
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return float((a_n * b_n).sum(axis=1).mean())


def train_mapper_with_holdout(z_plus, item_features, deg, behavior_dim=64, epochs=50, lr=1e-3, holdout_ratio=0.2):
    """Train mapper with item-level holdout split. Returns (model, pca, audit_dict)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_event("BEHAVIOR", "MAPPER_START", dim=behavior_dim, epochs=epochs, holdout_ratio=holdout_ratio)

    # ── PCA on ALL valid items ──
    valid_mask = z_plus.sum(axis=1) != 0
    valid_idx = np.where(valid_mask)[0]
    pca = PCA(n_components=behavior_dim, random_state=42)
    z_pca_all = np.zeros((z_plus.shape[0], behavior_dim), dtype=np.float32)
    z_pca_all[valid_idx] = pca.fit_transform(z_plus[valid_idx]).astype(np.float32)

    # ── Split items by degree tertiles ──
    train_idx, heldout_idx, strata = _stratify_by_degree(deg, valid_idx, holdout_ratio)

    # ── PCA(LLM) for degradation check ──
    pca_llm = PCA(n_components=behavior_dim, random_state=42)
    llm_pca_all = pca_llm.fit_transform(item_features[valid_idx].astype(np.float32))
    cos_z_true_vs_llm = _cosine_safe(z_pca_all[valid_idx], llm_pca_all)

    log_event("BEHAVIOR", "SPLIT",
              n_total=len(valid_idx), n_train=len(train_idx), n_heldout=len(heldout_idx),
              n_head=int((strata == 2).sum()), n_mid=int((strata == 1).sum()), n_tail=int((strata == 0).sum()),
              cos_z_true_vs_llm=cos_z_true_vs_llm)

    # ── Train mapper on train split only ──
    x_train = torch.from_numpy(item_features[train_idx]).float().to(device)
    y_train = torch.from_numpy(z_pca_all[train_idx]).float().to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 512), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, behavior_dim),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n_train = len(x_train)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for start in range(0, n_train, 2048):
            idx = perm[start: start + 2048]
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)
        if epoch % 10 == 0:
            with torch.no_grad():
                all_pred = model(x_train)
                cos_train = float(torch.nn.functional.cosine_similarity(all_pred, y_train, dim=1).mean())
            log_event("BEHAVIOR", "MAPPER_EPOCH", epoch=epoch, mse=total_loss / n_train, train_cos=cos_train)

    model.eval()

    # ── Evaluate on all splits ──
    def _eval_split(name, idx_list):
        if len(idx_list) == 0:
            return {"n": 0}
        x = torch.from_numpy(item_features[idx_list]).float().to(device)
        y_true = z_pca_all[idx_list]
        with torch.no_grad():
            y_pred = model(x).cpu().numpy()
        return {
            "n": len(idx_list),
            "cos": _cosine_safe(y_pred, y_true),
            "mse": float(np.mean((y_pred - y_true) ** 2)),
            "pred_std": float(y_pred.std(axis=0).mean()),
        }

    audit = {}
    audit["train"] = _eval_split("train", train_idx)
    audit["heldout"] = _eval_split("heldout", heldout_idx)

    # Per-stratum
    for s_name, s_val in [("head", 2), ("mid", 1), ("tail", 0)]:
        s_idx = valid_idx[strata == s_val]
        s_train = np.intersect1d(s_idx, train_idx)
        s_held = np.intersect1d(s_idx, heldout_idx)
        audit[f"{s_name}_train"] = _eval_split(f"{s_name}_train", s_train)
        audit[f"{s_name}_heldout"] = _eval_split(f"{s_name}_heldout", s_held)

    # Covariance effective rank of pred_z
    y_pred_all = np.zeros_like(z_pca_all)
    with torch.no_grad():
        for start in range(0, len(valid_idx), 4096):
            end = min(start + 4096, len(valid_idx))
            idx = valid_idx[start:end]
            y_pred_all[idx] = model(torch.from_numpy(item_features[idx]).float().to(device)).cpu().numpy()
    audit["cov_eff_rank"] = _covariance_effective_rank(y_pred_all[valid_idx])
    audit["cos_pred_vs_llm_pca"] = _cosine_safe(y_pred_all[valid_idx], llm_pca_all)
    audit["cos_true_vs_llm_pca"] = cos_z_true_vs_llm
    audit["explained_var"] = float(pca.explained_variance_ratio_.sum())
    audit["behavior_dim"] = behavior_dim
    audit["holdout_ratio"] = holdout_ratio
    audit["n_total_valid"] = int(valid_mask.sum())

    # Collapse / degradation flags
    audit["pred_collapsed"] = audit["train"]["pred_std"] < 0.01
    audit["degraded_to_semantic"] = audit["cos_pred_vs_llm_pca"] > 0.85

    log_event("BEHAVIOR", "AUDIT",
              train_cos=audit["train"]["cos"],
              heldout_cos=audit["heldout"]["cos"],
              head_heldout_cos=audit["head_heldout"]["cos"],
              mid_heldout_cos=audit["mid_heldout"]["cos"],
              tail_heldout_cos=audit["tail_heldout"]["cos"],
              cov_eff_rank=audit["cov_eff_rank"],
              cos_pred_vs_llm=audit["cos_pred_vs_llm_pca"],
              degraded=audit["degraded_to_semantic"],
              collapsed=audit["pred_collapsed"])

    return model.cpu(), pca, audit


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build AlphaFree-style behavior embeddings")
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--K-c", type=int, default=100)
    parser.add_argument("--behavior-dim", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--cooc-mode", default="log_count", choices=["log_count", "ppmi", "cosine"])
    parser.add_argument("--holdout-ratio", type=float, default=0.2,
                        help="Fraction of items held out per popularity stratum")
    parser.add_argument("--mapper-epochs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path("artifacts/behavior")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build behavior representations
    z_plus, item_features, deg = build_behavior_representations(
        args.source, args.data_root, K_c=args.K_c, tau=args.tau, cooc_mode=args.cooc_mode
    )

    # 2. Train mapper with holdout
    mapper, pca, audit = train_mapper_with_holdout(
        z_plus, item_features, deg, behavior_dim=args.behavior_dim,
        epochs=args.mapper_epochs, holdout_ratio=args.holdout_ratio
    )

    # 3. Save artifacts (unchanged from original)
    valid_mask = z_plus.sum(axis=1) != 0
    z64 = np.zeros((z_plus.shape[0], args.behavior_dim), dtype=np.float32)
    z64[valid_mask] = pca.transform(z_plus[valid_mask]).astype(np.float32)
    z64_path = out_dir / "source_behavior_z64.pt"
    torch.save(torch.from_numpy(z64), z64_path)
    log_event("ARTIFACT", "SAVED", path=str(z64_path))

    mapper_path = out_dir / "source_behavior_mapper.pt"
    torch.save({
        "mapper_state": mapper.state_dict(),
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "behavior_dim": args.behavior_dim,
    }, mapper_path)
    log_event("ARTIFACT", "SAVED", path=str(mapper_path))

    # 4. Save JSON audit report
    audit_path = out_dir / "source_behavior_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=float)
    log_event("ARTIFACT", "SAVED", path=str(audit_path))

    # 5. Final summary
    verdict = "PASS"
    issues = []
    if audit["heldout"]["cos"] < 0.7:
        verdict = "FAIL"; issues.append("heldout_cos < 0.7 → mapper not generalizing")
    if audit["degraded_to_semantic"]:
        verdict = "FAIL"; issues.append("pred_z degenerated to LLM PCA")
    if audit["pred_collapsed"]:
        verdict = "FAIL"; issues.append("mapper collapsed")
    if audit["cov_eff_rank"] < 5:
        issues.append(f"cov_eff_rank={audit['cov_eff_rank']} — pred_z low diversity")
    if audit["tail_heldout"]["cos"] < audit["head_heldout"]["cos"] * 0.5:
        issues.append("tail heldout cos much worse than head — unfair to cold-start items")

    log_event("BEHAVIOR", "VERDICT", verdict=verdict, issues=issues if issues else "none")
    log_event("BEHAVIOR", "DONE")


if __name__ == "__main__":
    main()
