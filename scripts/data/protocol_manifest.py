from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict zero-shot protocol guard. This no longer builds target "
            "causal popularity stores."
        )
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    paths = cfg["paths"]
    result_dir = Path(paths["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": "strict_source_only_temporal_v1",
        "source": str(cfg["source"]),
        "targets_declared_but_not_loaded": list(cfg["targets"]),
        "target_interactions_loaded_during_training": False,
        "target_catalog_interaction_statistics_built": False,
        "target_catalog_interaction_statistics_used": False,
        "target_prefix_timestamps_allowed_at_inference": True,
        "candidate_semantic_embeddings_allowed": True,
        "note": (
            "PrepRec-style target popularity stores are intentionally disabled. "
            "The temporal expert derives tokens on-the-fly from one user's "
            "observed prefix only."
        ),
    }
    output = result_dir / "strict_zero_shot_protocol_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
