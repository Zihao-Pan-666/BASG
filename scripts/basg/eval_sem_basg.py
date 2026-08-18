"""Evaluate SEM-based BASG across all 3 seeds (tuning + zero-shot eval).

Reuses existing TemporalExpert checkpoints — no retraining needed.
Only re-tunes the static fusion beta for the SEM backbone and evaluates.

Usage:
    python scripts/basg/eval_sem_basg.py              # all 3 seeds
    python scripts/basg/eval_sem_basg.py --seed 2026  # single seed

Memory ~5-10GB (inference-only); safe to run alongside other GPU workloads.
"""

from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import subprocess
import time
from datetime import datetime

PYTHON = sys.executable
CONFIG = "configs/basg/expert_sem.yaml"
TUNE_SCRIPT = "scripts/basg/tune_fusion.py"
EVAL_SCRIPT = "scripts/basg/eval_zero_shot.py"
SEEDS = [2026, 2027, 2028]
TARGETS = "amazon_cds_and_vinyl,amazon_industrial_and_scientific"


def run_direct(cmd: list[str], desc: str) -> bool:
    """Run a subprocess with stdout/stderr inherited so tqdm renders natively."""
    print(f"\n{'=' * 70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {desc}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'=' * 70}\n", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    m, s = divmod(int(elapsed), 60)
    ok = rc == 0
    status = "OK" if ok else f"FAILED (rc={rc})"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {desc} -> {status}  [{m}m{s:02d}s]\n", flush=True)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="SEM-based BASG: tune + eval all seeds")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    seeds = [args.seed] if args.seed else SEEDS

    print("=" * 70)
    print("  SEM-based BASG -- Tuning + Zero-Shot Evaluation")
    print(f"  Config: {CONFIG}")
    print(f"  Seeds:  {seeds}")
    print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70, flush=True)

    failed = 0
    total_start = time.perf_counter()

    for seed in seeds:
        # Step 1: Tune static beta on source validation
        ok = run_direct(
            [PYTHON, TUNE_SCRIPT, "--config", CONFIG, "--seed", str(seed)],
            f"TUNE  SEM+TE  seed={seed}",
        )
        if not ok:
            print(f"[SKIP EVAL] seed={seed} tuning failed", flush=True)
            failed += 1
            continue

        # Step 2: Evaluate on target domains
        ok = run_direct(
            [PYTHON, EVAL_SCRIPT, "--config", CONFIG, "--seed", str(seed),
             "--targets", TARGETS],
            f"EVAL  SEM+TE  seed={seed}",
        )
        if not ok:
            failed += 1

    total = time.perf_counter() - total_start
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    print("\n" + "=" * 70)
    print(f"  SEM-based BASG Complete")
    print(f"  Total: {h}h{m:02d}m{s:02d}s")
    print(f"  Failed: {failed}/{len(seeds)*2} stages")
    print(f"  Output: {_PROJECT_ROOT / 'results/mainline/sem_basg'}")
    print("=" * 70, flush=True)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
