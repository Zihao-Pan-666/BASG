from __future__ import annotations
from typing import List, Tuple

def build_source_train_val_samples(seqs: List[dict], max_len: int = 50, min_prefix: int = 1) -> Tuple[List[dict], List[dict]]:
    train, val = [], []
    for s in seqs:
        items, times = list(s["items"]), list(s["times"])
        if len(items) < max(min_prefix + 2, 3):
            continue
        for pos in range(min_prefix, len(items) - 1):
            train.append({"user": s["user"], "history": items[:pos][-max_len:], "history_times": times[:pos][-max_len:], "target": items[pos], "target_time": times[pos]})
        val.append({"user": s["user"], "history": items[:-1][-max_len:], "history_times": times[:-1][-max_len:], "target": items[-1], "target_time": times[-1]})
    return train, val
