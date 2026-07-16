"""Scoring entrypoint: python -m src.score --run outputs/<experiment_name>.

Categorical metrics use the subset of rows with a usable gold label AND a
valid (non-null) prediction. Dimensional metrics use the full split, minus
rows with missing gold VAD (25/session, unscoreable by construction) and
rows where the prediction failed even after retry.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from src.data import LABELS
from src.metrics import (
    accuracy,
    confusion_matrix,
    granularity_diagnostic,
    macro_f1,
    mae_per_dim,
    mean_ignore_nan,
    pearson_per_dim,
    per_class_f1,
    rmse_per_dim,
)


def load_preds(run_dir: Path) -> list[dict]:
    """Load preds.jsonl, deduping by utterance_id (last write wins).

    A resumed run appends a fresh attempt for any utterance_id whose prior
    attempt failed (see src/run.py's load_done_ids), so the same id can
    appear more than once -- the later record reflects the retry.
    """
    preds_path = run_dir / "preds.jsonl"
    records: dict[str, dict] = {}
    with open(preds_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["utterance_id"]] = rec
    return list(records.values())


def build_categorical_subset(records: list[dict]) -> tuple[list[str], list[str], dict]:
    gold, pred = [], []
    n_excluded = 0
    n_invalid = 0
    for rec in records:
        if rec["gold_label"] is None:
            n_excluded += 1
            continue
        if rec["pred_label"] is None:
            n_invalid += 1
            continue
        gold.append(rec["gold_label"])
        pred.append(rec["pred_label"])
    counts = {"n_scored": len(gold), "n_excluded": n_excluded, "n_invalid": n_invalid}
    return gold, pred, counts


def build_dimensional_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    preds, golds = [], []
    n_nan_gold_dropped = 0
    n_pred_invalid_dropped = 0
    for rec in records:
        gold_vad = rec["gold_vad"]
        if any(gold_vad[dim] is None for dim in ("v", "a", "d")):
            n_nan_gold_dropped += 1
            continue
        pred_vad = rec["pred_vad"]
        if any(pred_vad[dim] is None for dim in ("v", "a", "d")):
            n_pred_invalid_dropped += 1
            continue
        golds.append([gold_vad["v"], gold_vad["a"], gold_vad["d"]])
        preds.append([pred_vad["v"], pred_vad["a"], pred_vad["d"]])
    counts = {
        "n_total_rows": len(records),
        "n_nan_gold_dropped": n_nan_gold_dropped,
        "n_pred_invalid_dropped": n_pred_invalid_dropped,
        "n_scored": len(golds),
    }
    preds_arr = np.array(preds, dtype=float) if preds else np.zeros((0, 3))
    golds_arr = np.array(golds, dtype=float) if golds else np.zeros((0, 3))
    return preds_arr, golds_arr, counts


def compute_cost_block(records: list[dict]) -> dict | None:
    """Stage1/stage2 latency + fallback/skip rates, C2c ("llm_select") runs
    only. Returns None (block omitted) if no record carries stage1 fields."""
    if not any("stage1_latency_ms" in r for r in records):
        return None
    n = len(records)
    stage1_lat = [r["stage1_latency_ms"] for r in records if r.get("stage1_latency_ms") is not None]
    stage2_lat = [r["stage2_latency_ms"] for r in records if r.get("stage2_latency_ms") is not None]
    n_fallback = sum(1 for r in records if r.get("fallback"))
    n_skipped = sum(1 for r in records if r.get("stage1_skipped"))
    return {
        "n_records": n,
        "stage1_latency_ms": {
            "mean": statistics.mean(stage1_lat) if stage1_lat else float("nan"),
            "total": sum(stage1_lat),
        },
        "stage2_latency_ms": {
            "mean": statistics.mean(stage2_lat) if stage2_lat else float("nan"),
            "total": sum(stage2_lat),
        },
        "fallback_rate": n_fallback / n if n else float("nan"),
        "stage1_skipped_rate": n_skipped / n if n else float("nan"),
    }


def _recency_set(pool_size: int, n_sel: int) -> set[int]:
    return set(range(pool_size - n_sel, pool_size))


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return statistics.mean(xs), statistics.pstdev(xs)


def compute_selection_overlap(primary_records: list[dict], compare_run_dirs: list[Path]) -> dict:
    """Per-utterance Jaccard overlap between primary_records' selected_indices
    and (a) the recency set, (b) each compare_run_dirs' selected_indices,
    matched by utterance_id. Eligibility (pool_size > n_sel) is determined
    from the primary run's own records; works for any C2 run (a/b/c) as the
    primary, since only selected_indices/pool_size are required.
    """
    primary_by_id = {r["utterance_id"]: r for r in primary_records if "selected_indices" in r}
    eligible_ids = [
        uid for uid, r in primary_by_id.items() if r["pool_size"] > len(r["selected_indices"])
    ]

    recency_j = []
    for uid in eligible_ids:
        r = primary_by_id[uid]
        sel = set(r["selected_indices"])
        rec = _recency_set(r["pool_size"], len(sel))
        recency_j.append(_jaccard(sel, rec))
    mean_r, std_r = _mean_std(recency_j)

    result = {
        "n_eligible": len(eligible_ids),
        "vs_recency": {"mean_jaccard": mean_r, "std_jaccard": std_r, "n": len(recency_j)},
        "vs_compared_runs": {},
    }
    for other_dir in compare_run_dirs:
        other_by_id = {r["utterance_id"]: r for r in load_preds(other_dir) if "selected_indices" in r}
        pair_j = []
        for uid in eligible_ids:
            if uid not in other_by_id:
                continue
            sel_a = set(primary_by_id[uid]["selected_indices"])
            sel_b = set(other_by_id[uid]["selected_indices"])
            pair_j.append(_jaccard(sel_a, sel_b))
        mean_o, std_o = _mean_std(pair_j)
        result["vs_compared_runs"][other_dir.name] = {
            "mean_jaccard": mean_o,
            "std_jaccard": std_o,
            "n": len(pair_j),
        }
    return result


def _score_run(run_dir: Path) -> dict:
    records = load_preds(run_dir)

    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            run_meta = json.load(f)
        meta = {
            "experiment_name": run_meta.get("experiment_name"),
            "model": run_meta.get("model"),
            "condition": run_meta.get("config", {}).get("condition"),
        }

    gold, pred, cat_counts = build_categorical_subset(records)
    categorical = {
        **cat_counts,
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "per_class_f1": per_class_f1(gold, pred) if gold else {},
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(gold, pred) if gold else [],
        },
    }

    preds_arr, golds_arr, dim_counts = build_dimensional_arrays(records)
    if len(preds_arr):
        pearson = pearson_per_dim(preds_arr, golds_arr)
        rmse = rmse_per_dim(preds_arr, golds_arr)
        mae = mae_per_dim(preds_arr, golds_arr)
        granularity = granularity_diagnostic(preds_arr)
    else:
        pearson = {"v": float("nan"), "a": float("nan"), "d": float("nan")}
        rmse = {"v": float("nan"), "a": float("nan"), "d": float("nan")}
        mae = {"v": float("nan"), "a": float("nan"), "d": float("nan")}
        granularity = {"v": 0, "a": 0, "d": 0}

    dimensional = {
        **dim_counts,
        "pearson": {**pearson, "mean": mean_ignore_nan(pearson)},
        "rmse": {**rmse, "mean": mean_ignore_nan(rmse)},
        "mae": {**mae, "mean": mean_ignore_nan(mae)},
        "granularity": granularity,
    }

    result = {"run_meta_summary": meta, "categorical": categorical, "dimensional": dimensional}

    cost = compute_cost_block(records)
    if cost is not None:
        result["cost"] = cost

    return result


def score_run(
    run_dir: Path,
    eval_dir: Path = Path("eval"),
    compare_selections: list[Path] | None = None,
) -> dict:
    result = _score_run(run_dir)
    if compare_selections is not None:
        primary_records = load_preds(run_dir)
        result["selection_overlap"] = compute_selection_overlap(primary_records, compare_selections)
    out_dir = eval_dir / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def print_summary(result: dict) -> None:
    cat = result["categorical"]
    dim = result["dimensional"]
    print("=" * 60)
    print("CATEGORICAL")
    print(
        f"  n_scored={cat['n_scored']} n_excluded={cat['n_excluded']} "
        f"n_invalid={cat['n_invalid']}"
    )
    print(f"  accuracy={cat['accuracy']:.4f}  macro_f1={cat['macro_f1']:.4f}")
    print("  per-class F1:", {k: round(v, 4) for k, v in cat["per_class_f1"].items()})
    print("  confusion matrix (rows=gold, cols=pred):")
    labels = cat["confusion_matrix"]["labels"]
    print("       " + " ".join(f"{l:>6}" for l in labels))
    for label, row in zip(labels, cat["confusion_matrix"]["matrix"]):
        print(f"  {label:>4} " + " ".join(f"{v:>6}" for v in row))
    print("=" * 60)
    print("DIMENSIONAL")
    print(
        f"  n_total_rows={dim['n_total_rows']} n_nan_gold_dropped={dim['n_nan_gold_dropped']} "
        f"n_pred_invalid_dropped={dim['n_pred_invalid_dropped']} n_scored={dim['n_scored']}"
    )
    print(f"  pearson: {dim['pearson']}")
    print(f"  rmse:    {dim['rmse']}")
    print(f"  mae:     {dim['mae']}")
    print(f"  granularity (distinct predicted values): {dim['granularity']}")
    print("=" * 60)


def print_cost_summary(cost: dict) -> None:
    print("COST (C2c)")
    print(f"  n_records={cost['n_records']}")
    print(
        f"  stage1_latency_ms: mean={cost['stage1_latency_ms']['mean']:.1f} "
        f"total={cost['stage1_latency_ms']['total']:.1f}"
    )
    print(
        f"  stage2_latency_ms: mean={cost['stage2_latency_ms']['mean']:.1f} "
        f"total={cost['stage2_latency_ms']['total']:.1f}"
    )
    print(f"  fallback_rate={cost['fallback_rate']:.4f}  stage1_skipped_rate={cost['stage1_skipped_rate']:.4f}")
    print("=" * 60)


def print_selection_overlap_summary(overlap: dict) -> None:
    print("SELECTION OVERLAP")
    print(f"  n_eligible={overlap['n_eligible']}")
    vr = overlap["vs_recency"]
    print(f"  vs_recency: mean_jaccard={vr['mean_jaccard']:.4f} std={vr['std_jaccard']:.4f} n={vr['n']}")
    for name, stats in overlap["vs_compared_runs"].items():
        print(
            f"  vs_{name}: mean_jaccard={stats['mean_jaccard']:.4f} "
            f"std={stats['std_jaccard']:.4f} n={stats['n']}"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--eval-dir", default="eval")
    parser.add_argument(
        "--compare-selections",
        nargs="*",
        default=None,
        metavar="RUN_DIR",
        help="Other C2 run dirs to compare selected_indices against via Jaccard overlap; "
        "pass with no values to compute only the vs-recency overlap for --run itself.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    eval_dir = Path(args.eval_dir)
    compare_dirs = [Path(p) for p in args.compare_selections] if args.compare_selections is not None else None
    result = score_run(run_dir, eval_dir, compare_selections=compare_dirs)
    print_summary(result)
    if "cost" in result:
        print_cost_summary(result["cost"])
    if "selection_overlap" in result:
        print_selection_overlap_summary(result["selection_overlap"])
    print(f"Wrote {eval_dir / run_dir.name / 'metrics.json'}")


if __name__ == "__main__":
    main()
