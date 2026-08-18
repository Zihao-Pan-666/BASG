# basg/features/semantic.py strict replacement with safer embedding-column detection

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import ast
import numpy as np
import pandas as pd
import torch


def _parse_embedding_cell(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, list):
        arr = np.asarray(x)
    elif isinstance(x, str):
        arr = np.asarray(ast.literal_eval(x))
    else:
        arr = np.asarray(x)
    return arr.astype(np.float32)


def find_embedding_parquet(
    data_root: str | Path,
    domain: str,
    embedding_dir: str = "semantic_embeddings",
) -> Path:
    root = Path(data_root)
    candidates = [
        root / embedding_dir / f"{domain}_embedding_llama_fixed.parquet",
        root / embedding_dir / f"{domain}_embedding_llama.parquet",
        root / embedding_dir / f"{domain}_embedding_llama3.parquet",
        root / "semantic_embeddings_fixed" / f"{domain}_embedding_llama_fixed.parquet",
        root / "semantic_embeddings" / f"{domain}_embedding_llama_fixed.parquet",
        root / "semantic_embeddings" / f"{domain}_embedding_llama.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find embedding parquet for {domain}. Tried={candidates}")


def _looks_like_vector_column(df: pd.DataFrame, col: str, sample_size: int = 20) -> bool:
    values = df[col].dropna().head(sample_size).tolist()
    if not values:
        return False
    ok = 0
    for x in values:
        try:
            arr = _parse_embedding_cell(x)
            if arr.ndim == 1 and arr.shape[0] > 8:
                ok += 1
        except Exception:
            pass
    return ok > 0


def _select_vector_col(df: pd.DataFrame, requested: str, path: Path) -> str:
    """
    Pick the real embedding vector column.

    This avoids accidentally selecting scalar columns such as OldEmbeddingItemId
    just because their names contain the substring "embedding".
    """
    if requested in df.columns and _looks_like_vector_column(df, requested):
        return requested

    priority = [
        "item_text_embedding",
        "text_embedding",
        "semantic_embedding",
        "embedding_vector",
        "embeddings",
        "embedding",
    ]
    for col in priority:
        if col in df.columns and _looks_like_vector_column(df, col):
            return col

    candidates = [
        c for c in df.columns
        if "embedding" in c.lower() and _looks_like_vector_column(df, c)
    ]
    if candidates:
        return candidates[0]

    raise ValueError(
        f"No valid vector embedding column found in {path}. "
        f"requested={requested}, columns={list(df.columns)}. "
        "A valid embedding column should contain 1-D numeric vectors, e.g. item_text_embedding."
    )


def load_semantic_embeddings(
    data_root: str | Path,
    domain: str,
    item_map: Dict[str, int],
    vector_col: str = "item_text_embedding",
    embedding_dir: str = "semantic_embeddings",
    strict: bool = True,
) -> torch.Tensor:
    if item_map is None:
        raise ValueError("Formal BASG experiments require item_map; row-order loading is forbidden.")

    path = find_embedding_parquet(data_root, domain, embedding_dir=embedding_dir)
    df = pd.read_parquet(path)

    id_candidates = [
        c for c in df.columns
        if c.lower() in {"rawitemid", "raw_item_id", "itemid", "item_id", "asin", "product_id"}
    ]
    if not id_candidates:
        raise ValueError(f"{path} has no item id column. Row-order alignment is forbidden.")
    id_col = "RawItemId" if "RawItemId" in df.columns else id_candidates[0]

    vector_col = _select_vector_col(df, vector_col, path)

    raw_ids = df[id_col].astype(str).tolist()
    vectors = []
    bad_shape = []
    for rid, x in zip(raw_ids, df[vector_col].tolist()):
        vec = _parse_embedding_cell(x)
        if vec.ndim != 1 or vec.shape[0] <= 8:
            bad_shape.append((rid, tuple(vec.shape)))
            continue
        vectors.append(vec)

    if bad_shape:
        raise ValueError(
            f"{path}: selected vector_col={vector_col}, but {len(bad_shape)} rows are not valid vectors. "
            f"samples={bad_shape[:10]}"
        )

    dims = [int(v.shape[0]) for v in vectors]
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent embedding dims in {path}: {sorted(set(dims))}")
    dim = dims[0]

    emb_by_raw = {}
    dups = []
    for raw, vec in zip(raw_ids, vectors):
        if raw in emb_by_raw:
            dups.append(raw)
        emb_by_raw[raw] = np.asarray(vec, dtype=np.float32)
    if dups:
        raise ValueError(f"{path} has duplicated item ids: {dups[:20]}")

    table = np.zeros((max(item_map.values()) + 1, dim), dtype=np.float32)
    missing, nonfinite, zero = [], [], []

    for raw, idx in item_map.items():
        raw = str(raw)
        if raw not in emb_by_raw:
            missing.append(raw)
            continue
        vec = emb_by_raw[raw]
        if not np.isfinite(vec).all():
            nonfinite.append(raw)
            if strict:
                continue
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.linalg.norm(vec)) == 0.0:
            zero.append(raw)
        table[idx] = vec

    if missing:
        raise ValueError(f"{domain}: {len(missing)} current items miss embeddings. samples={missing[:20]}")
    if strict and nonfinite:
        raise ValueError(f"{domain}: {len(nonfinite)} embeddings contain NaN/Inf. samples={nonfinite[:20]}")
    if strict and zero:
        raise ValueError(f"{domain}: {len(zero)} embeddings are zero vectors. samples={zero[:20]}")

    print(
        f"[SemanticEmbedding] domain={domain} file={path} id_col={id_col} "
        f"vector_col={vector_col} table_shape={table.shape}",
        flush=True,
    )
    return torch.from_numpy(table)
