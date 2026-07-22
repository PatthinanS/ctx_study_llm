import json
import math

import pytest

from reccon_study.leakage import resolve_k
from reccon_study.selection_eval import (
    CAUSE_TYPES,
    aggregate_bucket,
    compute_target_stats,
    macro_prf1_with_ci,
    precision_recall_f1,
)


def test_precision_recall_f1_nan_on_zero_precision_denominator():
    result = precision_recall_f1(tp=0, n_scoreable_selected=0, n_gold_in_pool=3)
    assert math.isnan(result["precision"])
    assert result["recall"] == pytest.approx(0.0)
    assert math.isnan(result["f1"])


def test_precision_recall_f1_nan_on_zero_recall_denominator():
    result = precision_recall_f1(tp=0, n_scoreable_selected=2, n_gold_in_pool=0)
    assert math.isnan(result["recall"])
    assert math.isnan(result["f1"])


def test_precision_recall_f1_normal_case():
    result = precision_recall_f1(tp=2, n_scoreable_selected=4, n_gold_in_pool=2)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 / 3)


def _pool(n):
    return [{"utterance_id": f"p{i}"} for i in range(n)]


def _target_align(csv_pos, cause_csv_pos, cause_uids, types, uid="target1"):
    dists = [csv_pos - c for c in cause_csv_pos]
    return {
        "utterance_id": uid,
        "csv_pos": csv_pos,
        "cause_csv_pos": cause_csv_pos,
        "cause_utterance_ids": cause_uids,
        "cause_unresolved": 0,
        "has_latent_marker": False,
        "types": types,
        "max_cause_distance_csv": max(dists) if dists else -1,
    }


def test_compute_target_stats_main_and_visible_pool_variant():
    pool = _pool(6)  # p0..p5
    target_align = _target_align(csv_pos=6, cause_csv_pos=[2, 4], cause_uids=["p2", "p4"],
                                  types=["hybrid", "latent"])
    visible_uids = {"p0", "p1", "p2", "p3", "p4"}  # p5 invisible
    rec = {"utterance_id": "target1", "selected_indices": [2, 4, 5], "pool_size": 6}

    stats = compute_target_stats(rec, pool, target_align, visible_uids, k=3)

    assert stats["tp"] == 2
    assert stats["n_scoreable_selected"] == 2
    assert stats["n_invisible_selected"] == 1
    assert stats["n_selected"] == 3
    assert stats["R"] == 2
    assert stats["main"]["precision"] == pytest.approx(1.0)
    assert stats["main"]["recall"] == pytest.approx(1.0)
    # visible_pool_variant denominator is min(k, |pool ∩ visible|) = min(3, 5) = 3,
    # not n_scoreable_selected (2) -- the wasted pick on the invisible turn p5 costs it.
    assert stats["visible_pool_variant"]["precision"] == pytest.approx(2 / 3)
    assert stats["visible_pool_variant"]["recall"] == pytest.approx(1.0)
    assert stats["distance_bucket"] == ">k"  # max_cause_distance_csv=4 > k=3


def test_compute_target_stats_distance_bucket_within_k():
    pool = _pool(6)
    target_align = _target_align(csv_pos=6, cause_csv_pos=[5], cause_uids=["p5"], types=[])
    rec = {"utterance_id": "target1", "selected_indices": [5], "pool_size": 6}
    stats = compute_target_stats(rec, pool, target_align, visible_uids={"p5"}, k=4)
    assert stats["distance_bucket"] == "<=k"  # distance 1 <= k 4


def test_compute_target_stats_r_excludes_self_cause():
    pool = _pool(6)
    # cause_csv_pos includes a self-referential entry (== csv_pos, target's own position)
    # plus one genuine in-pool cause -- R must only count the latter.
    target_align = _target_align(csv_pos=6, cause_csv_pos=[6, 2], cause_uids=["target1", "p2"],
                                  types=[])
    rec = {"utterance_id": "target1", "selected_indices": [2], "pool_size": 6}
    stats = compute_target_stats(rec, pool, target_align, visible_uids={"p2"}, k=1)
    assert stats["R"] == 1


def test_compute_target_stats_derives_c1_selection():
    pool = _pool(6)
    target_align = _target_align(csv_pos=6, cause_csv_pos=[4], cause_uids=["p4"], types=[])
    rec = {"utterance_id": "target1"}  # C1-shaped: no selected_indices/pool_size
    stats = compute_target_stats(rec, pool, target_align, visible_uids={"p4", "p5"}, k=2)
    # derive_recency_selection(6, 2) -> [4, 5]
    assert stats["n_selected"] == 2
    assert stats["tp"] == 1  # p4 is the gold cause and was (recency-)selected


def test_macro_prf1_with_ci_drops_nan_targets():
    precisions = [1.0, float("nan"), 0.5]
    recalls = [1.0, 1.0, 0.5]
    f1s = [1.0, float("nan"), 0.5]
    result = macro_prf1_with_ci(precisions, recalls, f1s, n_resamples=200, seed=1)
    assert result["n"] == 3
    # macro mean over the two non-nan precision values: (1.0 + 0.5) / 2
    assert result["precision"] == pytest.approx(0.75)


def test_macro_prf1_with_ci_deterministic_by_seed():
    precisions = [1.0, 0.5, 0.0, 0.8]
    recalls = [1.0, 0.5, 0.0, 0.8]
    f1s = [1.0, 0.5, 0.0, 0.8]
    r1 = macro_prf1_with_ci(precisions, recalls, f1s, n_resamples=300, seed=3)
    r2 = macro_prf1_with_ci(precisions, recalls, f1s, n_resamples=300, seed=3)
    assert r1 == r2


def test_aggregate_bucket_empty_list_returns_nan_n_zero():
    result = aggregate_bucket([], "main", n_resamples=100, seed=1)
    assert result["n"] == 0
    assert math.isnan(result["precision"])
    assert math.isnan(result["unscoreable_rate"])


def _fake_stat(types, distance_bucket, precision=1.0):
    return {
        "main": {"precision": precision, "recall": precision, "f1": precision},
        "visible_pool_variant": {"precision": precision, "recall": precision, "f1": precision},
        "n_selected": 2, "n_scoreable_selected": 2, "n_invisible_selected": 0, "R": 1,
        "types": types, "distance_bucket": distance_bucket,
    }


def test_multi_membership_type_bucket_assignment():
    stats_list = [
        _fake_stat(["hybrid", "latent"], "<=k"),
        _fake_stat(["no-context"], ">k"),
    ]
    by_type = {
        t: aggregate_bucket([s for s in stats_list if t in s["types"]], "main", n_resamples=50, seed=1)
        for t in CAUSE_TYPES
    }
    assert by_type["hybrid"]["n"] == 1
    assert by_type["latent"]["n"] == 1
    assert by_type["no-context"]["n"] == 1
    assert by_type["inter-personal"]["n"] == 0
    assert by_type["self-contagion"]["n"] == 0


def test_distance_bucket_split():
    stats_list = [
        _fake_stat([], "<=k"),
        _fake_stat([], "<=k"),
        _fake_stat([], ">k"),
    ]
    by_distance = {
        b: aggregate_bucket([s for s in stats_list if s["distance_bucket"] == b], "main", n_resamples=50, seed=1)
        for b in ["<=k", ">k"]
    }
    assert by_distance["<=k"]["n"] == 2
    assert by_distance[">k"]["n"] == 1


def test_resolve_k_from_run_meta_config_context_k(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "preds.jsonl").write_text("")
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({"config": {"context": {"k": 7}}}, f)
    assert resolve_k(run_dir) == 7


def test_resolve_k_override_takes_priority(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "preds.jsonl").write_text("")
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({"config": {"context": {"k": 7}}}, f)
    assert resolve_k(run_dir, override=9) == 9


def test_resolve_k_c2_fallback_from_records(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    with open(run_dir / "preds.jsonl", "w") as f:
        f.write(json.dumps({"utterance_id": "u1", "selected_indices": [0, 1, 2], "pool_size": 10}) + "\n")
        f.write(json.dumps({"utterance_id": "u2", "selected_indices": [0], "pool_size": 1}) + "\n")
    assert resolve_k(run_dir) == 3


def test_resolve_k_raises_for_c1_without_run_meta(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    with open(run_dir / "preds.jsonl", "w") as f:
        f.write(json.dumps({"utterance_id": "u1", "condition": "C1"}) + "\n")
    with pytest.raises(ValueError):
        resolve_k(run_dir)
