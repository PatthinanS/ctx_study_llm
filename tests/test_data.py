from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    EXCLUDED_EMOTIONS,
    LABELS,
    build_context,
    is_categorical_usable,
    load_iemocap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = REPO_ROOT / "data" / "iemocap" / "iemocap_merged_all.csv"


def test_build_context_window_last_k(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    result = build_context("window", target, history, k=3)
    assert result == synthetic_dialogue[2:5]
    assert [t["utterance_id"] for t in result] == [
        t["utterance_id"] for t in synthetic_dialogue[2:5]
    ]


def test_build_context_empty_first_turn(synthetic_dialogue):
    target = synthetic_dialogue[0]
    result = build_context("window", target, [], k=3)
    assert result == []


def test_build_context_none_strategy_always_empty(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    assert build_context("none", target, history, k=3) == []


def test_build_context_unknown_strategy_raises(synthetic_dialogue):
    with pytest.raises(ValueError):
        build_context("bogus", synthetic_dialogue[5], synthetic_dialogue[:5], k=3)


def test_is_categorical_usable_true_for_labels():
    for label in LABELS:
        assert is_categorical_usable({"emotion": label}) is True


def test_is_categorical_usable_false_for_excluded():
    for emotion in EXCLUDED_EMOTIONS:
        assert is_categorical_usable({"emotion": emotion}) is False


def test_is_categorical_usable_false_for_nan():
    assert is_categorical_usable({"emotion": float("nan")}) is False


def test_context_includes_excluded_label_rows(synthetic_dialogue):
    # turn index 3 has emotion="xxx" -- excluded from categorical scoring
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    result = build_context("window", target, history, k=3)
    assert any(t["emotion"] == "xxx" for t in result)


def test_get_splits_reads_config_directly():
    from src.data import get_splits

    cfg = {
        "splits": {
            "test_session": "Session5",
            "val_session": "Session4",
            "train_sessions": ["Session1", "Session2", "Session3"],
        }
    }
    train, val, test = get_splits(cfg)
    assert train == ["Session1", "Session2", "Session3"]
    assert val == "Session4"
    assert test == "Session5"


def test_get_splits_overlap_raises():
    from src.data import get_splits

    cfg = {
        "splits": {
            "test_session": "Session5",
            "val_session": "Session4",
            "train_sessions": ["Session1", "Session4"],
        }
    }
    with pytest.raises(ValueError):
        get_splits(cfg)


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real IEMOCAP CSV not present in data/")
def test_session5_split_row_count():
    df = load_iemocap(str(REAL_CSV))
    session5 = df[df["session"] == "Session5"]
    assert len(session5) == 2195
