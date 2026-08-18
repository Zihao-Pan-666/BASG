from __future__ import annotations

from typing import List, Sequence, Set
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_left(seq: Sequence[int], max_len: int, pad_id: int = 0) -> list[int]:
    seq = list(seq)[-max_len:]
    return [pad_id] * (max_len - len(seq)) + seq


class PrefixDataset(Dataset):
    def __init__(self, samples: List[dict], num_items: int, max_len: int = 50):
        self.samples = samples
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        hist = pad_left(s["history"], self.max_len, 0)
        return {
            "history": torch.tensor(hist, dtype=torch.long),
            "target": torch.tensor(int(s["target"]), dtype=torch.long),
            "target_time": torch.tensor(float(s.get("target_time", 0.0)), dtype=torch.float32),
        }


class NegativeSampler:
    def __init__(self, num_items: int, seed: int = 2026):
        self.num_items = int(num_items)
        self.rng = random.Random(seed)

    def sample(self, forbidden: Set[int], n: int) -> list[int]:
        out = []
        max_tries = max(100, n * 50)
        tries = 0
        while len(out) < n and tries < max_tries:
            x = self.rng.randint(1, self.num_items)
            if x not in forbidden:
                out.append(x)
            tries += 1
        while len(out) < n:
            out.append(self.rng.randint(1, self.num_items))
        return out


def collate_prefix(batch):
    return {
        "history": torch.stack([b["history"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "target_time": torch.stack([b["target_time"] for b in batch]),
    }
