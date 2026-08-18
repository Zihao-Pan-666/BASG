from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


_VECTOR_COLUMNS = (
    "item_text_embedding",
    "embedding",
    "vector",
    "text_embedding",
    "semantic_embedding",
)


def _parse_vector(value: Any) -> np.ndarray:
    """Parse one embedding cell into a finite float32 vector."""
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty embedding string")
        try:
            arr = np.asarray(ast.literal_eval(text))
        except (ValueError, SyntaxError) as exc:
            raise ValueError("unable to parse embedding string") from exc
    else:
        arr = np.asarray(value)

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty embedding vector")
    if not np.isfinite(arr).all():
        raise ValueError("embedding vector contains NaN/Inf")
    return arr


def _detect_vector_column(frame: pd.DataFrame) -> str:
    for column in _VECTOR_COLUMNS:
        if column in frame.columns:
            return column

    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        try:
            vec = _parse_vector(series.iloc[0])
        except (TypeError, ValueError):
            continue
        if vec.size >= 8:
            return str(column)

    raise KeyError(
        "Could not detect the embedding vector column. "
        f"Available columns: {list(frame.columns)}"
    )


def find_catalog_embedding_file(
    data_root: str | Path,
    embedding_dir: str,
    domain: str,
) -> Path:
    """Locate a domain embedding parquet without reading domain interactions."""
    root = Path(data_root)
    base = root / embedding_dir
    candidates = (
        base / f"{domain}_embedding_llama_fixed.parquet",
        base / f"{domain}_embedding_llama.parquet",
        base / f"{domain}_embedding_llama3.parquet",
        base / f"{domain}_embedding.parquet",
        base / f"{domain}.parquet",
    )
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(base.glob(f"*{domain}*.parquet"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No semantic embedding parquet found for domain={domain!r} under {base}"
        )
    raise RuntimeError(
        f"Multiple semantic embedding files found for domain={domain!r}: "
        + ", ".join(str(p) for p in matches)
    )


def load_catalog_embedding_pool(
    data_root: str | Path,
    embedding_dir: str,
    domain: str,
    pool_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    Load only item-side semantic metadata for an auxiliary domain.

    No auxiliary interactions are opened. A deterministic catalog subset is
    selected before vectors are stacked, limiting memory used by RecG/SAGE.
    """
    path = find_catalog_embedding_file(data_root, embedding_dir, domain)
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Empty embedding table: {path}")

    vector_col = _detect_vector_column(frame)
    valid_rows = frame[vector_col].notna().to_numpy().nonzero()[0]
    if valid_rows.size == 0:
        raise ValueError(f"No non-null vectors in {path}:{vector_col}")

    if pool_size > 0 and valid_rows.size > pool_size:
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(valid_rows, size=int(pool_size), replace=False))
    else:
        chosen = valid_rows

    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for row_index in chosen.tolist():
        vec = _parse_vector(frame.iloc[row_index][vector_col])
        if expected_dim is None:
            expected_dim = int(vec.size)
        elif vec.size != expected_dim:
            raise ValueError(
                f"Inconsistent vector dimension in {path}: "
                f"expected {expected_dim}, got {vec.size} at row {row_index}"
            )
        vectors.append(vec)

    matrix = torch.from_numpy(np.stack(vectors).astype(np.float32, copy=False))
    metadata = {
        "domain": domain,
        "path": str(path),
        "vector_col": vector_col,
        "catalog_rows": int(len(frame)),
        "pool_rows": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "interaction_labels_used": False,
    }
    return matrix, metadata
