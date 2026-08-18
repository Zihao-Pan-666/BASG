from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


def _pad_left_int(values: Sequence[int], max_len: int, pad_value: int = 0) -> list[int]:
    values = list(values)[-max_len:]
    return [pad_value] * (max_len - len(values)) + values


def _pad_left_float64(
    values: Sequence[float], max_len: int, pad_value: float = 0.0
) -> list[float]:
    """
    Preserve millisecond Unix timestamps in float64.

    Float32 cannot accurately retain small differences around timestamps of
    order 1e12, which silently corrupts inter-arrival-gap features.
    """
    values = list(values)[-max_len:]
    return [pad_value] * (max_len - len(values)) + values


class PopDynPrefixDataset(Dataset):
    """
    Prefix dataset with per-interaction timestamps.

    It contains only the same per-user prefix and next-item label already used
    by the existing BERT4Rec protocol. It does not compute or retain any
    catalog-level interaction statistics.
    """

    def __init__(self, samples: list[dict], num_items: int, max_len: int = 50) -> None:
        self.samples = samples
        self.num_items = int(num_items)
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        history = list(sample["history"])
        history_times = list(sample.get("history_times", [0.0] * len(history)))
        if len(history_times) != len(history):
            raise ValueError("history and history_times must have equal length.")

        return {
            "history": torch.tensor(
                _pad_left_int(history, self.max_len), dtype=torch.long
            ),
            "history_times": torch.tensor(
                _pad_left_float64(history_times, self.max_len), dtype=torch.float64
            ),
            "target": torch.tensor(int(sample["target"]), dtype=torch.long),
            "target_time": torch.tensor(
                float(sample.get("target_time", 0.0)), dtype=torch.float64
            ),
        }


def collate_popdyn_prefix(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "history": torch.stack([item["history"] for item in batch]),
        "history_times": torch.stack([item["history_times"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "target_time": torch.stack([item["target_time"] for item in batch]),
    }
