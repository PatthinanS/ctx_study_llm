"""Per-run leakage analysis: classifies every C1/C2-selected turn against the
RECCON alignment as gold_cause / visible_noncause / invisible, and compares
the observed invisible-selection rate against the random-pool-composition
baseline.

Pool reconstruction is CSV-derived (via `src.data.reconstruct_dialogues`),
never read off the preds.jsonl record's own fields except as a cross-check --
this is what lets C1 (which stores neither `selected_indices` nor `pool_size`
at all) and C2 (which stores both) share the same code path. See
`reccon_study/README.md` and the design-decision comments in `align.py` for
why pool ordering must come from `src.data`, not an independently-derived one.

Only "scoreable"-bucket targets (`reccon_study.eligibility`) are scored; see
that module's docstring for why the other four buckets would otherwise
silently distort every rate reported here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reccon_study.cli import (
    ALL_SESSIONS,
    OUT_DIR,
    add_runs_arg,
    add_sessions_arg,
    banner,
    bootstrap_ci,
    kv_line,
    parse_sessions,
    section,
    write_report,
)
from reccon_study.eligibility import BUCKETS, classify_eligibility
from src.data import load_iemocap, reconstruct_dialogues
from src.score import load_preds

DEFAULT_ALIGNMENT = str(OUT_DIR / "reccon_ie_aligned.json")
DEFAULT_CSV = "data/iemocap/iemocap_merged_all.csv"


def classify_selection(selected_uid: str, cause_uids: set[str], visible_uids: set[str]) -> str:
    """"gold_cause" | "visible_noncause" | "invisible"."""
    if selected_uid in cause_uids:
        return "gold_cause"
    if selected_uid in visible_uids:
        return "visible_noncause"
    return "invisible"


def build_utterance_index(df, sessions: list[str] | None = None) -> dict[str, tuple[str, Any, int]]:
    """utterance_id -> (session, dialog, idx_in_dialogue). idx_in_dialogue IS
    the pool size for that utterance (pool = every prior turn of its
    dialogue), from the exact same ordering `src/run.py` used at inference
    time (`reconstruct_dialogues`, group by (session,dialog), sort by
    start_time)."""
    if sessions is None:
        sessions = ALL_SESSIONS
    dialogues = reconstruct_dialogues(df, sessions=sessions)
    index: dict[str, tuple[str, Any, int]] = {}
    for (session, dialog), rows in dialogues.items():
        for idx, row in enumerate(rows):
            index[row["utterance_id"]] = (session, dialog, idx)
    return index


def derive_recency_selection(pool_size: int, k: int) -> list[int]:
    """C1's selection isn't stored -- it's always the last n_sel=min(k,pool_size)
    pool indices (plain recency), matching src/score.py's `_recency_set`."""
    n_sel = min(k, pool_size)
    return list(range(pool_size - n_sel, pool_size))


def get_selection(rec: dict, pool_size: int, k: int) -> list[int]:
    """Real selection for C2 records (asserting the reconstructed pool size
    matches the record's own stored `pool_size` -- the off-by-one canary);
    derived recency selection for C1 records, which carry neither
    `selected_indices` nor `pool_size`."""
    if "selected_indices" in rec:
        stored_pool_size = rec.get("pool_size")
        if stored_pool_size is not None and stored_pool_size != pool_size:
            raise AssertionError(
                f"pool_size mismatch for utterance_id={rec.get('utterance_id')!r}: "
                f"reconstructed from CSV={pool_size} vs stored in preds.jsonl={stored_pool_size}. "
                "This is the off-by-one canary -- a real mismatch here means the CSV-derived "
                "dialogue ordering no longer matches the ordering src/run.py used at inference time."
            )
        return rec["selected_indices"]
    return derive_recency_selection(pool_size, k)


def resolve_k(run_dir: Path, override: int | None = None) -> int:
    """`--k` override if given; else `run_meta.json`'s logged config
    (`config.context.k`); else (C2 runs only) infer from the max observed
    `len(selected_indices)` among records where the pool wasn't smaller than
    k. C1 runs with no `run_meta.json` have no signal to infer k from at all
    (their records carry neither field) -- raises rather than guessing."""
    if override is not None:
        return override
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        k = meta.get("config", {}).get("context", {}).get("k")
        if k is not None:
            return k
    records = load_preds(run_dir)
    candidates = [
        len(r["selected_indices"]) for r in records
        if "selected_indices" in r and r.get("pool_size", 0) > len(r["selected_indices"])
    ]
    if candidates:
        return max(candidates)
    raise ValueError(
        f"Cannot resolve k for {run_dir}: no run_meta.json with config.context.k, and no "
        "C2 selected_indices to infer it from. If this is a C1 run without run_meta.json, "
        "pass --k explicitly."
    )


def paired_bootstrap_diff(
    observed: list[float], baseline: list[float], n_resamples: int = 10000, seed: int = 42
) -> dict:
    """Per-target paired differences (observed_i - baseline_i), bootstrap-
    resampled at the TARGET level (not per-selection -- a target's several
    selections aren't independent draws). Returns
    {"diff": mean, "ci_lo":..., "ci_hi":...}; all nan if either list is empty."""
    observed_arr = np.array(observed, dtype=float)
    baseline_arr = np.array(baseline, dtype=float)
    n = len(observed_arr)

    def diff_stat(idx: np.ndarray) -> float:
        return float(np.mean(observed_arr[idx] - baseline_arr[idx]))

    point, lo, hi = bootstrap_ci(n, diff_stat, n_resamples, seed)
    return {"diff": point, "ci_lo": lo, "ci_hi": hi}


def build_alignment_lookup(alignment: dict) -> tuple[dict[str, dict], dict[str, set[str]]]:
    """target_by_uid: utterance_id -> its alignment utterance-record.
    visible_uids_by_dialog: dialog name -> set of every utterance_id RECCON
    aligned in that dialogue."""
    target_by_uid: dict[str, dict] = {}
    visible_uids_by_dialog: dict[str, set[str]] = {}
    for dialog, record in alignment.items():
        visible_uids_by_dialog[dialog] = {u["utterance_id"] for u in record["utterances"]}
        for u in record["utterances"]:
            target_by_uid[u["utterance_id"]] = u
    return target_by_uid, visible_uids_by_dialog


def run_leakage(
    run_dirs: list[Path],
    alignment: dict,
    csv_path: str,
    sessions: list[str],
    seed: int = 42,
    n_resamples: int = 10000,
    k_override: int | None = None,
) -> dict:
    df = load_iemocap(csv_path)
    dialogues = reconstruct_dialogues(df, sessions=ALL_SESSIONS)
    utt_index = build_utterance_index(df, sessions=ALL_SESSIONS)
    _target_by_uid, visible_uids_by_dialog = build_alignment_lookup(alignment)

    scoped_targets = [
        (dialog, u)
        for dialog, record in alignment.items()
        if record["session"] in sessions
        for u in record["utterances"]
    ]

    report: dict[str, Any] = {"_meta": {"sessions": sessions, "seed": seed, "n_resamples": n_resamples}}

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        records = load_preds(run_dir)
        records_by_uid = {r["utterance_id"]: r for r in records}
        k = resolve_k(run_dir, k_override)

        counts = {"gold_cause": 0, "visible_noncause": 0, "invisible": 0}
        n_selections = 0
        observed_rates: list[float] = []
        baseline_rates: list[float] = []
        funnel = {b: 0 for b in BUCKETS}

        for dialog, target_align in scoped_targets:
            uid = target_align["utterance_id"]
            in_preds = uid in records_by_uid
            session, dialog_key, idx = utt_index[uid]
            pool_size = idx
            bucket = classify_eligibility(target_align, in_preds, pool_size, k)
            funnel[bucket] += 1
            if bucket != "scoreable":
                continue

            rec = records_by_uid[uid]
            pool = dialogues[(session, dialog_key)][:idx]
            selection = get_selection(rec, pool_size, k)

            visible_uids = visible_uids_by_dialog[dialog]
            cause_uids = set(target_align["cause_utterance_ids"])

            n_invisible_selected = 0
            for i in selection:
                selected_uid = pool[i]["utterance_id"]
                cls = classify_selection(selected_uid, cause_uids, visible_uids)
                counts[cls] += 1
                n_selections += 1
                if cls == "invisible":
                    n_invisible_selected += 1
            observed_rates.append(n_invisible_selected / len(selection))

            n_invisible_in_pool = sum(1 for r in pool if r["utterance_id"] not in visible_uids)
            baseline_rates.append(n_invisible_in_pool / pool_size)

        funnel["aligned"] = len(scoped_targets)
        total = sum(counts.values())
        rates = {cls: (v / total if total else float("nan")) for cls, v in counts.items()}

        n_targets = len(observed_rates)
        diff_result = paired_bootstrap_diff(observed_rates, baseline_rates, n_resamples, seed)
        observed_mean = float(np.mean(observed_rates)) if n_targets else float("nan")
        baseline_mean = float(np.mean(baseline_rates)) if n_targets else float("nan")
        diff_lo, diff_hi = diff_result["ci_lo"], diff_result["ci_hi"]

        if n_targets == 0:
            interp = "no scoreable targets -- nothing to compare"
        elif diff_hi < 0:
            interp = "BELOW random baseline (diff CI excludes 0) -- incomplete-judgment bias appears small"
        elif diff_lo > 0:
            interp = "ABOVE random baseline (diff CI excludes 0) -- selector prefers invisible turns more than chance"
        else:
            interp = "not distinguishable from random baseline (diff CI includes 0)"

        report[run_dir.name] = {
            "funnel": funnel,
            "n_selections": n_selections,
            "counts": counts,
            "rates": rates,
            "vs_random_baseline": {
                "observed_invisible_rate": observed_mean,
                "baseline_invisible_rate": baseline_mean,
                "diff": diff_result["diff"],
                "diff_ci_lo": diff_lo,
                "diff_ci_hi": diff_hi,
                "n_resamples": n_resamples,
                "seed": seed,
                "n_targets": n_targets,
            },
            "interpretation": interp,
        }

    return report


def print_leakage_report(report: dict) -> None:
    for run_name, data in report.items():
        if run_name == "_meta":
            continue
        banner(f"LEAKAGE: {run_name}")
        f = data["funnel"]
        section("funnel:")
        kv_line(2, aligned=f["aligned"], not_in_preds=f["not_in_preds"], no_cause=f["no_cause"],
                no_in_pool_cause=f["no_in_pool_cause"], forced_selection=f["forced_selection"],
                scoreable=f["scoreable"])
        section("selection classification (pooled over scoreable targets' selections):")
        kv_line(2, n_selections=data["n_selections"], **data["counts"])
        kv_line(2, **{f"rate_{cls}": v for cls, v in data["rates"].items()})
        vb = data["vs_random_baseline"]
        section(f"vs random baseline (macro, target-level bootstrap, n={vb['n_targets']}):")
        kv_line(2, observed_invisible_rate=vb["observed_invisible_rate"],
                baseline_invisible_rate=vb["baseline_invisible_rate"])
        kv_line(2, diff=vb["diff"], ci_lo=vb["diff_ci_lo"], ci_hi=vb["diff_ci_hi"])
        print(f"  {data['interpretation']}")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runs_arg(parser)
    add_sessions_arg(parser)
    parser.add_argument("--alignment", default=DEFAULT_ALIGNMENT)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=None,
                         help="Override k for all --runs (needed for a C1 run with no run_meta.json).")
    args = parser.parse_args()

    with open(args.alignment) as f:
        alignment = json.load(f)
    sessions = parse_sessions(args.sessions)
    run_dirs = [Path(p) for p in args.runs]

    report = run_leakage(
        run_dirs, alignment, args.csv, sessions,
        seed=args.seed, n_resamples=args.n_resamples, k_override=args.k,
    )
    print_leakage_report(report)
    path = write_report("leakage", report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
