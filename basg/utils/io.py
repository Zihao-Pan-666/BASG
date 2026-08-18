from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import torch


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def append_csv(row: dict, path):
    """Append a single row to a CSV file (pandas-based, column-aligned)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # Align columns explicitly to prevent schema drift.
        cols = list(dict.fromkeys(list(old.columns) + list(df.columns)))
        old = old.reindex(columns=cols)
        df = df.reindex(columns=cols)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def append_csv_dictwriter(path: str | Path, row: dict, fieldnames: list[str] | None = None):
    """Append a single row to a CSV using csv.DictWriter (lightweight, no pandas).

    On first write, the header is written using *fieldnames* (or row keys).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fieldnames if fieldnames is not None else list(row.keys())
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Atomically save a PyTorch checkpoint (write to .tmp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def torch_load_safe(path: str | Path, map_location: str = "cpu", weights_only: bool = False):
    """Load a PyTorch checkpoint with standard safety settings."""
    return torch.load(str(path), map_location=map_location, weights_only=weights_only)
