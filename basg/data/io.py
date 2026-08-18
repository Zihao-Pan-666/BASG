from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import ast
import numpy as np
import pandas as pd
import torch


USER_COLS = {"userid", "user_id", "user", "reviewerid", "reviewer_id"}
ITEM_COLS = {"itemid", "item_id", "item", "asin", "product_id"}
TIME_COLS = {"timestamp", "time", "unixreviewtime", "unix_review_time"}


def find_processed_csv(data_root: str | Path, domain: str) -> Path:
    root = Path(data_root)
    candidates = [
        root / "processed" / f"{domain}.csv",
        root / domain / "processed_data.csv",
        root / domain / f"{domain}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find processed csv for {domain}. Tried: {candidates}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in USER_COLS:
            rename[c] = "UserId"
        elif lc in ITEM_COLS:
            rename[c] = "ItemId"
        elif lc in TIME_COLS:
            rename[c] = "Timestamp"
    df = df.rename(columns=rename)
    required = {"UserId", "ItemId", "Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}; available={list(df.columns)}")
    out = df[["UserId", "ItemId", "Timestamp"]].copy()
    out["UserId"] = out["UserId"].astype(str)
    out["ItemId"] = out["ItemId"].astype(str)
    out["Timestamp"] = pd.to_numeric(out["Timestamp"], errors="coerce").fillna(0).astype(float)
    return out


def build_item_map(df: pd.DataFrame, start_index: int = 1) -> Dict[str, int]:
    # Stable mapping by first appearance after chronological sorting.
    seen: Dict[str, int] = {}
    next_id = start_index
    for x in df["ItemId"].tolist():
        if x not in seen:
            seen[x] = next_id
            next_id += 1
    return seen


def load_interactions(data_root: str | Path, domain: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    path = find_processed_csv(data_root, domain)
    df = normalize_columns(pd.read_csv(path))
    df = df.sort_values(["UserId", "Timestamp"]).reset_index(drop=True)
    item_map = build_item_map(df, start_index=1)
    df["RawItemId"] = df["ItemId"]
    df["ItemId"] = df["ItemId"].map(item_map).astype(int)
    return df, item_map


def group_user_sequences(df: pd.DataFrame, min_len: int = 3) -> List[dict]:
    seqs = []
    for user, g in df.sort_values(["UserId", "Timestamp"]).groupby("UserId", sort=False):
        items = g["ItemId"].astype(int).tolist()
        times = g["Timestamp"].astype(float).tolist()
        if len(items) >= min_len:
            seqs.append({"user": user, "items": items, "times": times})
    return seqs


def split_source_prefix_samples(seqs: List[dict], max_len: int = 50, min_prefix: int = 1) -> List[dict]:
    samples = []
    for s in seqs:
        items, times = s["items"], s["times"]
        # Leave the last item for source validation by default; training scripts can choose split externally.
        for pos in range(min_prefix, len(items) - 1):
            hist = items[:pos][-max_len:]
            hist_times = times[:pos][-max_len:]
            target = items[pos]
            target_time = times[pos]
            samples.append({
                "user": s["user"], "history": hist, "history_times": hist_times,
                "target": target, "target_time": target_time,
            })
    return samples


def build_target_eval_samples(seqs: List[dict], max_len: int = 50) -> List[dict]:
    samples = []
    for s in seqs:
        items, times = s["items"], s["times"]
        if len(items) < 2:
            continue
        samples.append({
            "user": s["user"],
            "history": items[:-1][-max_len:],
            "history_times": times[:-1][-max_len:],
            "target": items[-1],
            "target_time": times[-1],
        })
    return samples
