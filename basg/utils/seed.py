from __future__ import annotations

import random
import re

import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_path_seed_override(cfg: dict, new_seed: int) -> None:
    """Override data.seed and replace seed in all path values.

    Used by scripts 15/16/17 to support --seed CLI flag.  Replaces
    occurrences of the old seed number (e.g. 'seed2026') with the new
    one in every string-valued path entry.
    """
    old = str(cfg.get("data", {}).get("seed", ""))
    cfg["data"]["seed"] = int(new_seed)
    if old:
        for key, value in cfg.get("paths", {}).items():
            if isinstance(value, str):
                cfg["paths"][key] = re.sub(
                    rf"seed{old}", f"seed{new_seed}", value
                )
