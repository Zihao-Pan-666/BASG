from __future__ import annotations

import hashlib


def partition_validation_samples_by_user(
    samples: list[dict],
    *,
    popdyn_selection_fraction: float,
    fusion_selection_fraction: float,
    salt: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Disjoint deterministic source-validation user partitions.

    popdyn selection: early stopping / checkpoint selection;
    fusion selection: static beta selection;
    calibration: reserved third source-validation partition (unused by the
    deployable BASG pipeline; retained for protocol documentation).

    Splitting by user avoids the same user's neighboring prefixes leaking across
    the three source-only decisions.
    """
    first = float(popdyn_selection_fraction)
    second = float(fusion_selection_fraction)
    if not (0.0 < first < 1.0 and 0.0 < second < 1.0 and first + second < 1.0):
        raise ValueError(
            "popdyn_selection_fraction and fusion_selection_fraction must be "
            "positive and sum to less than 1."
        )

    popdyn_selection: list[dict] = []
    fusion_selection: list[dict] = []
    calibration: list[dict] = []
    for sample in samples:
        digest = hashlib.sha256(f"{salt}|{sample['user']}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "little") / float(2**64)
        if value < first:
            popdyn_selection.append(sample)
        elif value < first + second:
            fusion_selection.append(sample)
        else:
            calibration.append(sample)

    if not popdyn_selection or not fusion_selection or not calibration:
        raise RuntimeError(
            "A source validation partition is empty. Adjust fractions or the salt."
        )
    return popdyn_selection, fusion_selection, calibration
