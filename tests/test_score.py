import json

from src.score import build_categorical_subset, build_dimensional_arrays, load_preds, score_run

RECORDS = [
    {
        "utterance_id": "u1",
        "gold_label": "ang",
        "pred_label": "ang",
        "gold_vad": {"v": 3.0, "a": 3.0, "d": 3.0},
        "pred_vad": {"v": 3.0, "a": 3.0, "d": 3.0},
    },
    {
        # excluded-label row (gold_label None): not a categorical target,
        # but still fully usable for dimensional scoring.
        "utterance_id": "u2",
        "gold_label": None,
        "pred_label": "hap",
        "gold_vad": {"v": 2.0, "a": 2.0, "d": 2.0},
        "pred_vad": {"v": 2.0, "a": 2.0, "d": 2.0},
    },
    {
        # model failed to produce a valid label/vad even after retry.
        "utterance_id": "u3",
        "gold_label": "hap",
        "pred_label": None,
        "gold_vad": {"v": 4.0, "a": 4.0, "d": 4.0},
        "pred_vad": {"v": None, "a": None, "d": None},
    },
    {
        # missing gold VAD (the ~25/session NaN-annotation rows).
        "utterance_id": "u4",
        "gold_label": "neu",
        "pred_label": "neu",
        "gold_vad": {"v": None, "a": None, "d": None},
        "pred_vad": {"v": 3.0, "a": 3.0, "d": 3.0},
    },
]


def test_load_preds_dedupes_keeping_last_occurrence(tmp_path):
    """A resumed run appends a fresh attempt for a previously-failed
    utterance_id (see load_done_ids in src/run.py); scoring must use the
    later (retried) record, not double-count both."""
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    with open(run_dir / "preds.jsonl", "w") as f:
        f.write(json.dumps({"utterance_id": "u1", "pred_label": None}) + "\n")
        f.write(json.dumps({"utterance_id": "u1", "pred_label": "hap"}) + "\n")
        f.write(json.dumps({"utterance_id": "u2", "pred_label": "neu"}) + "\n")

    records = load_preds(run_dir)

    assert len(records) == 2
    by_id = {r["utterance_id"]: r for r in records}
    assert by_id["u1"]["pred_label"] == "hap"
    assert by_id["u2"]["pred_label"] == "neu"


def test_categorical_subset_excludes_null_gold_and_null_pred():
    gold, pred, counts = build_categorical_subset(RECORDS)
    assert gold == ["ang", "neu"]
    assert pred == ["ang", "neu"]
    assert counts == {"n_scored": 2, "n_excluded": 1, "n_invalid": 1}


def test_dimensional_subset_drops_nan_gold_vad():
    preds, golds, counts = build_dimensional_arrays(RECORDS)
    assert len(preds) == 2  # u1, u2
    assert counts == {
        "n_total_rows": 4,
        "n_nan_gold_dropped": 1,
        "n_pred_invalid_dropped": 1,
        "n_scored": 2,
    }


def test_score_run_writes_metrics_json_structure(tmp_path):
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    with open(run_dir / "preds.jsonl", "w") as f:
        for rec in RECORDS:
            f.write(json.dumps(rec) + "\n")

    eval_dir = tmp_path / "eval"
    result = score_run(run_dir, eval_dir)

    metrics_path = eval_dir / "fake_run" / "metrics.json"
    assert metrics_path.exists()
    assert not (run_dir / "metrics.json").exists()
    assert set(result) == {"run_meta_summary", "categorical", "dimensional"}
    assert set(result["categorical"]) >= {
        "n_scored",
        "n_excluded",
        "n_invalid",
        "accuracy",
        "macro_f1",
        "per_class_f1",
        "confusion_matrix",
    }
    assert set(result["dimensional"]) >= {
        "n_total_rows",
        "n_nan_gold_dropped",
        "n_pred_invalid_dropped",
        "n_scored",
        "pearson",
        "rmse",
        "mae",
        "granularity",
    }

    with open(metrics_path) as f:
        on_disk = json.load(f)
    assert on_disk == result
