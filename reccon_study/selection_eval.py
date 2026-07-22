"""Per-run selection-quality scoring against RECCON gold causes: condensed-
list precision/recall/F1 (Sakai & Kando 2008) over the scoreable subset, a
visible-pool counterfactual variant, and breakdowns by RECCON cause type and
cause-distance bucket. C1 doubles as the recency floor here -- include a C1
run directory in --runs to get it as an ordinary entry, directly comparable
to the C2 entries, rather than a synthetic baseline computed separately.

No bpref: with no rank order surviving into preds.jsonl (`src/run.py` always
stores selections pre-sorted chronologically ascending), a ranked bpref
can't be computed, and the natural unranked-set treatment (Buckley &
Voorhees 2004's bracket) reduces algebraically to plain recall -- it carries
no information the scoreable-subset metric here doesn't already report. See
`reccon_study/README.md` for the full citation/rationale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    small_n_marker,
    write_report,
)
from reccon_study.eligibility import BUCKETS, classify_eligibility, in_pool_R
from reccon_study.leakage import (
    build_alignment_lookup,
    build_utterance_index,
    classify_selection,
    get_selection,
    resolve_k,
)
from src.data import load_iemocap, reconstruct_dialogues
from src.score import load_preds

DEFAULT_ALIGNMENT = str(OUT_DIR / "reccon_ie_aligned.json")
DEFAULT_CSV = "data/iemocap/iemocap_merged_all.csv"

CAUSE_TYPES = ["no-context", "inter-personal", "self-contagion", "hybrid", "latent"]


def precision_recall_f1(tp: int, n_scoreable_selected: int, n_gold_in_pool: int) -> dict:
    """nan on zero denominators, mirroring src/metrics.py's per_class_f1/
    macro_f1 nan-on-empty convention. This IS the condensed-list metric --
    invisible/unjudged selections are already excluded from
    `n_scoreable_selected` upstream, no separate step needed."""
    precision = tp / n_scoreable_selected if n_scoreable_selected else float("nan")
    recall = tp / n_gold_in_pool if n_gold_in_pool else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_target_stats(rec: dict, pool: list[dict], target_align: dict, visible_uids: set[str], k: int) -> dict:
    """Only ever called for a "scoreable"-bucket target (R >= 1 guaranteed).
    `rec` may be a real C2 record or a C1 record (get_selection dispatches)."""
    pool_size = len(pool)
    selection = get_selection(rec, pool_size, k)
    cause_uids = set(target_align["cause_utterance_ids"])
    R = in_pool_R(target_align)

    tp = n_scoreable_selected = n_invisible_selected = 0
    for i in selection:
        uid = pool[i]["utterance_id"]
        cls = classify_selection(uid, cause_uids, visible_uids)
        if cls == "gold_cause":
            tp += 1
            n_scoreable_selected += 1
        elif cls == "visible_noncause":
            n_scoreable_selected += 1
        else:
            n_invisible_selected += 1
    main = precision_recall_f1(tp, n_scoreable_selected, R)

    # Visible-pool variant: a proxy for "what if the candidate pool had only
    # ever contained visible turns". We can't re-run the actual selector
    # under that constraint, so the numerator stays the model's real
    # visible-cause hits (tp), but the denominator becomes the visible-only
    # pool's own selection budget (min(k, |pool ∩ visible|)) instead of the
    # count of turns it actually picked -- this penalizes picks spent on
    # invisible turns as wasted budget, rather than simply ignoring them
    # (which is what `main` already does).
    pool_v_size = sum(1 for r in pool if r["utterance_id"] in visible_uids)
    n_sel_v = min(k, pool_v_size)
    visible_pool_variant = precision_recall_f1(tp, n_sel_v, R)

    dist = target_align["max_cause_distance_csv"]
    distance_bucket = "<=k" if 0 <= dist <= k else ">k"

    return {
        "utterance_id": target_align["utterance_id"],
        "tp": tp,
        "n_scoreable_selected": n_scoreable_selected,
        "n_invisible_selected": n_invisible_selected,
        "n_selected": len(selection),
        "R": R,
        "main": main,
        "visible_pool_variant": visible_pool_variant,
        "types": target_align["types"],
        "distance_bucket": distance_bucket,
    }


def macro_prf1_with_ci(
    precisions: list[float], recalls: list[float], f1s: list[float],
    n_resamples: int = 10000, seed: int = 42,
) -> dict:
    """Macro (per-target, then averaged) precision/recall/F1 with a 95%
    percentile bootstrap CI (target-level resampling). nan per-target values
    (e.g. a target whose entire selection was invisible has undefined
    precision) are dropped from each resample's mean, not treated as 0."""
    p_arr = np.array(precisions, dtype=float)
    r_arr = np.array(recalls, dtype=float)
    f_arr = np.array(f1s, dtype=float)
    n = len(p_arr)

    def mk_stat(arr: np.ndarray):
        def stat(idx: np.ndarray) -> float:
            vals = arr[idx]
            valid = vals[~np.isnan(vals)]
            return float(valid.mean()) if len(valid) else float("nan")
        return stat

    p_point, p_lo, p_hi = bootstrap_ci(n, mk_stat(p_arr), n_resamples, seed)
    r_point, r_lo, r_hi = bootstrap_ci(n, mk_stat(r_arr), n_resamples, seed)
    f_point, f_lo, f_hi = bootstrap_ci(n, mk_stat(f_arr), n_resamples, seed)
    return {
        "precision": p_point, "precision_ci": [p_lo, p_hi],
        "recall": r_point, "recall_ci": [r_lo, r_hi],
        "f1": f_point, "f1_ci": [f_lo, f_hi],
        "n": n,
    }


def aggregate_bucket(
    stats_list: list[dict], metric_key: str, n_resamples: int = 10000, seed: int = 42
) -> dict:
    """Macro-aggregates one metric_key ("main" or "visible_pool_variant")
    across a list of compute_target_stats() dicts, adding count fields."""
    precisions = [s[metric_key]["precision"] for s in stats_list]
    recalls = [s[metric_key]["recall"] for s in stats_list]
    f1s = [s[metric_key]["f1"] for s in stats_list]
    result = macro_prf1_with_ci(precisions, recalls, f1s, n_resamples, seed)
    result["n_selected"] = sum(s["n_selected"] for s in stats_list)
    result["n_scoreable_selected"] = sum(s["n_scoreable_selected"] for s in stats_list)
    result["n_gold_in_pool"] = sum(s["R"] for s in stats_list)
    n_inv = sum(s["n_invisible_selected"] for s in stats_list)
    n_sel_total = sum(s["n_selected"] for s in stats_list)
    result["unscoreable_rate"] = n_inv / n_sel_total if n_sel_total else float("nan")
    return result


def run_selection_eval(
    run_dirs: list[Path],
    alignment: dict,
    csv_path: str,
    sessions: list[str],
    n_resamples: int = 10000,
    seed: int = 42,
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

    report: dict = {"_meta": {"sessions": sessions, "n_resamples": n_resamples, "seed": seed}}

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        records = load_preds(run_dir)
        records_by_uid = {r["utterance_id"]: r for r in records}
        k = resolve_k(run_dir, k_override)

        funnel = {b: 0 for b in BUCKETS}
        target_stats = []
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
            visible_uids = visible_uids_by_dialog[dialog]
            target_stats.append(compute_target_stats(rec, pool, target_align, visible_uids, k))
        funnel["aligned"] = len(scoped_targets)

        by_type = {
            t: aggregate_bucket([s for s in target_stats if t in s["types"]], "main", n_resamples, seed)
            for t in CAUSE_TYPES
        }
        by_distance = {
            b: aggregate_bucket([s for s in target_stats if s["distance_bucket"] == b], "main", n_resamples, seed)
            for b in ["<=k", ">k"]
        }

        report[run_dir.name] = {
            "k": k,
            "funnel": funnel,
            "main": aggregate_bucket(target_stats, "main", n_resamples, seed),
            "visible_pool_variant": aggregate_bucket(target_stats, "visible_pool_variant", n_resamples, seed),
            "by_type": by_type,
            "by_distance": by_distance,
        }

    return report


def print_selection_eval_report(report: dict) -> None:
    def print_block(name: str, block: dict) -> None:
        marker = small_n_marker(block["n"])
        section(f"{name} (n={block['n']}{marker}):")
        kv_line(4, precision=block["precision"], recall=block["recall"], f1=block["f1"])
        kv_line(4, precision_ci=block["precision_ci"], recall_ci=block["recall_ci"], f1_ci=block["f1_ci"])
        if "unscoreable_rate" in block:
            kv_line(4, unscoreable_rate=block["unscoreable_rate"], n_selected=block["n_selected"],
                    n_scoreable_selected=block["n_scoreable_selected"], n_gold_in_pool=block["n_gold_in_pool"])

    for run_name, data in report.items():
        if run_name == "_meta":
            continue
        banner(f"SELECTION EVAL: {run_name} (k={data['k']})")
        f = data["funnel"]
        section("funnel:")
        kv_line(2, aligned=f["aligned"], not_in_preds=f["not_in_preds"], no_cause=f["no_cause"],
                no_in_pool_cause=f["no_in_pool_cause"], forced_selection=f["forced_selection"],
                scoreable=f["scoreable"])
        print_block("main", data["main"])
        print_block("visible_pool_variant", data["visible_pool_variant"])
        section("by_type:")
        for t, block in data["by_type"].items():
            print_block(f"  {t}", block)
        section("by_distance:")
        for b, block in data["by_distance"].items():
            print_block(f"  {b}", block)
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runs_arg(parser)
    add_sessions_arg(parser)
    parser.add_argument("--alignment", default=DEFAULT_ALIGNMENT)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=None,
                         help="Override k for all --runs (needed for a C1 run with no run_meta.json).")
    args = parser.parse_args()

    with open(args.alignment) as f:
        alignment = json.load(f)
    sessions = parse_sessions(args.sessions)
    run_dirs = [Path(p) for p in args.runs]

    report = run_selection_eval(
        run_dirs, alignment, args.csv, sessions,
        n_resamples=args.n_resamples, seed=args.seed, k_override=args.k,
    )
    print_selection_eval_report(report)
    path = write_report("selection_eval", report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
