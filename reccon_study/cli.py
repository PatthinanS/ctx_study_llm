"""Shared CLI/report plumbing for align.py, leakage.py, selection_eval.py.

Not a `python -m reccon_study.cli` entry point itself -- each of the three
modules has its own `main()` and imports what it needs from here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

ALL_SESSIONS = ["Session1", "Session2", "Session3", "Session4", "Session5"]
OUT_DIR = Path("outputs/reccon")


def parse_sessions(raw: str | None) -> list[str]:
    """'4,5' -> ['Session4','Session5']. None/empty -> ALL_SESSIONS (a copy)."""
    if not raw:
        return list(ALL_SESSIONS)
    return [f"Session{tok.strip()}" for tok in raw.split(",") if tok.strip()]


def write_report(name: str, payload: dict, out_dir: Path = OUT_DIR) -> Path:
    """json.dump(payload, indent=2) to out_dir/<name>.json; mkdir -p; returns path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def banner(title: str) -> None:
    print("=" * 60)
    print(title)


def section(title: str) -> None:
    print(title)


def kv_line(indent: int = 2, **kwargs) -> None:
    parts = []
    for key, value in kwargs.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    print(" " * indent + " ".join(parts))


def small_n_marker(n: int, threshold: int = 30) -> str:
    return " [SMALL N]" if n < threshold else ""


def add_sessions_arg(parser) -> None:
    parser.add_argument(
        "--sessions",
        default=None,
        help='Comma-separated session numbers, e.g. "4,5". Default: all 5 (all 16 '
        "RECCON-annotated dialogues) -- valid for any C0-C2 prompting-only condition, "
        "since no IEMOCAP training happens in this repo.",
    )


def add_runs_arg(parser) -> None:
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="RUN_DIR",
        help="One or more outputs/<experiment_name> run directories (C1 or C2).",
    )


def bootstrap_ci(
    n_items: int, statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int = 10000, seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile (2.5, 97.5) bootstrap over `n_items` indices, resampled with
    replacement. `statistic_fn(idx_array) -> float` is called once on the
    observed (identity) index array for the point estimate, then once per
    resample for the CI. Generic over "mean of a per-target value list" (pass
    `lambda idx: values[idx].mean()`) and "paired per-target difference" (pass
    `lambda idx: (observed[idx] - baseline[idx]).mean()`) so both leakage.py's
    and selection_eval.py's bootstraps share one implementation.

    Returns (point, ci_lo, ci_hi); all nan if n_items == 0.
    """
    if n_items == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n_items)
    point = statistic_fn(idx_all)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample_idx = rng.integers(0, n_items, size=n_items)
        stats[i] = statistic_fn(resample_idx)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)
