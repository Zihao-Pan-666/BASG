from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import torch

from basg.data.dataset import PrefixDataset
from basg.data.io import group_user_sequences, load_interactions


def set_seed_compat(seed: int) -> None:
    """Support both current and historical seed helper names."""
    try:
        from basg.utils.seed import set_global_seed
        set_global_seed(int(seed))
        return
    except ImportError:
        pass

    from basg.utils.seed import set_seed
    set_seed(int(seed))


def resolve_device_compat(name: str) -> torch.device:
    """Support both current and historical device helper names."""
    try:
        from basg.utils.device import resolve_device
        return resolve_device(name)
    except ImportError:
        from basg.utils.device import get_device
        return get_device(name)


def load_interactions_compat(
    data_root: str,
    domain: str,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Normalize old `(frame, item_map)` and newer
    `(frame, user_map, item_map)` loaders.
    """
    result = load_interactions(data_root, domain)
    if not isinstance(result, tuple):
        raise TypeError("load_interactions must return a tuple")

    if len(result) == 3:
        frame, user_map, item_map = result
    elif len(result) == 2:
        frame, item_map = result
        if "UserId" not in frame.columns:
            raise KeyError("Interaction frame has no UserId column")
        unique_users = frame["UserId"].astype(str).drop_duplicates().tolist()
        user_map = {user: index for index, user in enumerate(unique_users)}
    else:
        raise TypeError(
            f"Unsupported load_interactions return length: {len(result)}"
        )
    return frame, user_map, item_map


def group_user_sequences_compat(frame: pd.DataFrame, min_len: int) -> list[dict]:
    signature = inspect.signature(group_user_sequences)
    kwargs: dict[str, Any] = {}
    if "min_len" in signature.parameters:
        kwargs["min_len"] = int(min_len)
    return group_user_sequences(frame, **kwargs)


def build_source_splits_compat(
    sequences: list[dict],
    max_len: int,
    min_len: int,
) -> tuple[list[dict], list[dict]]:
    from basg.data.semantic_splits import build_source_train_val_samples

    signature = inspect.signature(build_source_train_val_samples)
    kwargs: dict[str, Any] = {}
    parameters = signature.parameters

    if "sequences" in parameters:
        kwargs["sequences"] = sequences
    elif "seqs" in parameters:
        kwargs["seqs"] = sequences
    else:
        # Positional fallback for an unknown but compatible historical name.
        kwargs[next(iter(parameters))] = sequences

    if "max_len" in parameters:
        kwargs["max_len"] = int(max_len)
    if "min_len" in parameters:
        kwargs["min_len"] = int(min_len)
    if "min_prefix" in parameters:
        kwargs["min_prefix"] = 1

    return build_source_train_val_samples(**kwargs)


def build_target_samples_compat(
    sequences: list[dict],
    max_len: int,
    min_len: int,
) -> list[dict]:
    try:
        from basg.data.semantic_splits import build_target_eval_samples
    except ImportError:
        from basg.data.io import build_target_eval_samples

    signature = inspect.signature(build_target_eval_samples)
    kwargs: dict[str, Any] = {}
    parameters = signature.parameters

    if "sequences" in parameters:
        kwargs["sequences"] = sequences
    elif "seqs" in parameters:
        kwargs["seqs"] = sequences
    else:
        kwargs[next(iter(parameters))] = sequences

    if "max_len" in parameters:
        kwargs["max_len"] = int(max_len)
    if "min_len" in parameters:
        kwargs["min_len"] = int(min_len)

    return build_target_eval_samples(**kwargs)


def make_prefix_dataset(
    samples: list[dict],
    num_items: int,
    max_len: int,
) -> PrefixDataset:
    """Normalize old/new PrefixDataset constructor signatures."""
    signature = inspect.signature(PrefixDataset)
    kwargs: dict[str, Any] = {"samples": samples}

    if "num_items" in signature.parameters:
        kwargs["num_items"] = int(num_items)
    if "max_len" in signature.parameters:
        kwargs["max_len"] = int(max_len)

    return PrefixDataset(**kwargs)
