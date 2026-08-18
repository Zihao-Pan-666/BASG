from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - the project still runs without tqdm
    tqdm = None


_TRUE_VALUES = {"1", "true", "yes", "on", "show", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "hide", "disabled", "quiet", "batch"}


def parse_bool(value: object, default: bool = False) -> bool:
    """Parse a permissive boolean value used by YAML and CLI compatibility code."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_progress_enabled(explicit: object | None = None) -> bool:
    """
    Resolve progress visibility with one project-wide precedence rule.

    BASG_PROGRESS=0 is useful for non-interactive batch launchers. Direct runs
    otherwise show progress by default.
    """
    environment = os.getenv("BASG_PROGRESS")
    if environment is not None:
        return parse_bool(environment)
    return parse_bool(explicit, default=True)


def resolve_runtime_config(
    config: dict[str, Any] | None,
    *,
    no_progress: bool = False,
    print_json: bool = False,
    progress_refresh: float | None = None,
) -> dict[str, Any]:
    """Merge YAML defaults and common CLI overrides into a compact runtime config."""
    cfg = dict(config or {})
    refresh = (
        float(progress_refresh)
        if progress_refresh is not None
        else float(cfg.get("progress_refresh_seconds", 0.5))
    )
    if refresh <= 0:
        raise ValueError("progress_refresh_seconds must be positive.")
    return {
        "show_progress": False
        if no_progress
        else resolve_progress_enabled(cfg.get("show_progress", True)),
        "print_json": bool(print_json or parse_bool(cfg.get("print_json", False))),
        "progress_refresh_seconds": refresh,
        "leave_progress_bar": parse_bool(cfg.get("leave_progress_bar", True), True),
        "save_timing_chart": parse_bool(cfg.get("save_timing_chart", False)),
    }


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "--"
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}" if abs(value) < 10 else f"{value:.3f}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    return str(value)


def log_event(phase: str, event: str, **fields: object) -> None:
    """Print one compact, machine-readable, project-wide status line."""
    prefix = f"[BASG][{phase.upper()}][{event.upper()}]"
    message = " ".join(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    print(f"{prefix} {message}".rstrip(), flush=True)


@dataclass
class RunClock:
    """Shared elapsed/ETA clock for epoch- and target-level summaries."""

    started_at: float = 0.0

    def __post_init__(self) -> None:
        if self.started_at <= 0:
            self.started_at = time.perf_counter()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.started_at)

    def eta(self, completed: int, total: int) -> float | None:
        if completed <= 0 or total <= completed:
            return 0.0 if total <= completed else None
        return self.elapsed() / float(completed) * float(total - completed)


def loader_total(loader: Iterable, max_batches: int = 0) -> int | None:
    try:
        total = int(len(loader))  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return int(max_batches) if max_batches > 0 else None
    return min(total, int(max_batches)) if max_batches > 0 else total


def progress_iter(
    iterable: Iterable,
    *,
    total: int | None,
    run_name: str,
    phase: str,
    detail: str = "",
    enabled: bool = True,
    refresh_seconds: float = 0.5,
    leave: bool = False,
) -> Iterator:
    """
    Return an iterable with a time-aware tqdm bar.

    tqdm renders completed/total, elapsed time, ETA, and throughput. The
    standard prefix makes training, tuning, and evaluation logs visually
    consistent in PowerShell, terminals, and log files.
    """
    if not enabled or tqdm is None:
        return iter(iterable)
    suffix = f" {detail}" if detail else ""
    return iter(
        tqdm(
            iterable,
            total=total,
            desc=f"[BASG][{phase.upper()}] {run_name}{suffix}",
            unit="batch",
            dynamic_ncols=True,
            leave=bool(leave),
            mininterval=float(refresh_seconds),
        )
    )


def progress_bar(
    *,
    total: int,
    run_name: str,
    phase: str,
    detail: str = "",
    enabled: bool = True,
    refresh_seconds: float = 0.5,
    leave: bool = False,
):
    """Create a manually-updated, time-aware progress bar or return None."""
    if not enabled or tqdm is None:
        return None
    suffix = f" {detail}" if detail else ""
    return tqdm(
        total=int(total),
        desc=f"[BASG][{phase.upper()}] {run_name}{suffix}",
        unit="step",
        dynamic_ncols=True,
        leave=bool(leave),
        mininterval=float(refresh_seconds),
    )


def set_progress_postfix(iterator: object, **fields: object) -> None:
    """Set a progress postfix when the iterator is a tqdm bar."""
    if hasattr(iterator, "set_postfix"):
        clean = {key: _format_value(value) for key, value in fields.items()}
        iterator.set_postfix(clean, refresh=False)  # type: ignore[attr-defined]


def close_progress(iterator: object) -> None:
    if hasattr(iterator, "close"):
        iterator.close()  # type: ignore[attr-defined]


def maybe_print_json(payload: object, *, enabled: bool) -> None:
    if enabled:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)


def save_training_timing_chart(
    history: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
) -> str | None:
    """
    Save a compact timing chart without making matplotlib a hard runtime dependency.

    The chart contains epoch duration and cumulative elapsed minutes. It is a
    sidecar diagnostic; failures to render never invalidate an experiment.
    """
    if not history:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional visual output
        log_event("RUNTIME", "CHART_SKIPPED", reason=type(error).__name__)
        return None

    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    epoch_seconds = [
        float(row.get("epoch_seconds", row.get("train_seconds", 0.0)))
        for row in history
    ]
    cumulative_minutes = [
        float(row.get("elapsed_seconds", 0.0)) / 60.0 for row in history
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.plot(epochs, epoch_seconds, marker="o", label="Epoch duration (s)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Epoch duration (s)")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis_right = axis.twinx()
    axis_right.plot(epochs, cumulative_minutes, marker="s", linestyle="--", label="Cumulative elapsed (min)")
    axis_right.set_ylabel("Cumulative elapsed (min)")
    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis.legend(handles + right_handles, labels + right_labels, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def save_stage_timing_chart(
    stages: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
) -> str | None:
    """Save a compact one-figure stage-duration chart for tuning/evaluation runs."""
    if not stages:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional visual output
        log_event("RUNTIME", "CHART_SKIPPED", reason=type(error).__name__)
        return None

    labels = [str(stage.get("label", f"stage_{index + 1}")) for index, stage in enumerate(stages)]
    seconds = [float(stage.get("elapsed_seconds", 0.0)) for stage in stages]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(6.0, 1.1 * len(labels)), 4.2))
    axis.bar(labels, seconds)
    axis.set_xlabel("Stage")
    axis.set_ylabel("Elapsed time (s)")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)
