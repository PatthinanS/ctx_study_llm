from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data as data_mod
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


# ---------------------------------------------------------------------------
# C2a: "random" strategy
# ---------------------------------------------------------------------------


def test_strategy_random_deterministic_by_seed_and_utterance_id(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    r1 = build_context("random", target, history, k=3, seed=42)
    r2 = build_context("random", target, history, k=3, seed=42)
    assert [t["utterance_id"] for t in r1] == [t["utterance_id"] for t in r2]


def test_strategy_random_independent_of_call_interleaving(synthetic_dialogue):
    """Simulates ThreadPoolExecutor interleaving: A's result must not depend
    on whether a call for a different utterance B happened in between."""
    history = synthetic_dialogue[:5]
    target_a = {**synthetic_dialogue[5], "utterance_id": "u_a"}
    target_b = {**synthetic_dialogue[5], "utterance_id": "u_b"}

    r_a1 = build_context("random", target_a, history, k=3, seed=42)
    r_b1 = build_context("random", target_b, history, k=3, seed=42)
    r_a2 = build_context("random", target_a, history, k=3, seed=42)
    r_b2 = build_context("random", target_b, history, k=3, seed=42)

    assert [t["utterance_id"] for t in r_a1] == [t["utterance_id"] for t in r_a2]
    assert [t["utterance_id"] for t in r_b1] == [t["utterance_id"] for t in r_b2]


def test_strategy_random_different_utterance_id_generally_differs(synthetic_dialogue):
    history = synthetic_dialogue[:5]
    results = []
    for uid in ["u_100", "u_101", "u_102", "u_103"]:
        target = {**synthetic_dialogue[5], "utterance_id": uid}
        r = build_context("random", target, history, k=3, seed=42)
        results.append(tuple(t["utterance_id"] for t in r))
    assert len(set(results)) > 1


def test_strategy_random_selected_indices_chronological_order(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    result = build_context("random", target, history, k=3, seed=42)
    times = [t["start_time"] for t in result]
    assert times == sorted(times)


def test_strategy_random_pool_smaller_than_k_returns_all(synthetic_dialogue):
    target = synthetic_dialogue[2]
    history = synthetic_dialogue[:2]
    result = build_context("random", target, history, k=5, seed=42)
    assert {t["utterance_id"] for t in result} == {t["utterance_id"] for t in history}


def test_strategy_random_requires_seed_kwarg(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    with pytest.raises(ValueError):
        build_context("random", target, history, k=3)


# ---------------------------------------------------------------------------
# C2b: "sim" strategy (encoder mocked via monkeypatching _encode_batch --
# no real torch/transformers call, no network)
# ---------------------------------------------------------------------------


def test_strategy_sim_selects_most_similar_turn(synthetic_dialogue, monkeypatch):
    data_mod._EMBED_CACHE.clear()
    target = synthetic_dialogue[3]
    history = synthetic_dialogue[:3]

    # cosine-to-target([1,0]): turn2 (~0.980) > turn0 (~0.707) > turn1 (~0.0995)
    vectors = {
        f"{history[0]['speaker']}: {history[0]['text']}": np.array([1.0, 1.0]),
        f"{history[1]['speaker']}: {history[1]['text']}": np.array([1.0, 10.0]),
        f"{history[2]['speaker']}: {history[2]['text']}": np.array([1.0, 0.2]),
        f"{target['speaker']}: {target['text']}": np.array([1.0, 0.0]),
    }

    def fake_encode(texts, encoder_name):
        return np.array([vectors[t] for t in texts])

    monkeypatch.setattr(data_mod, "_encode_batch", fake_encode)

    result = build_context("sim", target, history, k=1)
    assert len(result) == 1
    assert result[0]["utterance_id"] == history[2]["utterance_id"]


def test_strategy_sim_reorders_chronologically(synthetic_dialogue, monkeypatch):
    data_mod._EMBED_CACHE.clear()
    target = synthetic_dialogue[3]
    history = synthetic_dialogue[:3]

    # similarity rank (desc): turn2, turn0, turn1 -- reverse of chronological
    # for the top 2 (turn2 then turn0); output must still be [turn0, turn2].
    vectors = {
        f"{history[0]['speaker']}: {history[0]['text']}": np.array([1.0, 1.0]),
        f"{history[1]['speaker']}: {history[1]['text']}": np.array([1.0, 10.0]),
        f"{history[2]['speaker']}: {history[2]['text']}": np.array([1.0, 0.2]),
        f"{target['speaker']}: {target['text']}": np.array([1.0, 0.0]),
    }

    def fake_encode(texts, encoder_name):
        return np.array([vectors[t] for t in texts])

    monkeypatch.setattr(data_mod, "_encode_batch", fake_encode)

    result = build_context("sim", target, history, k=2)
    assert [t["utterance_id"] for t in result] == [
        history[0]["utterance_id"],
        history[2]["utterance_id"],
    ]


def test_strategy_sim_caches_embeddings_across_calls(synthetic_dialogue, monkeypatch):
    data_mod._EMBED_CACHE.clear()
    calls = []

    def fake_encode(texts, encoder_name):
        calls.append(list(texts))
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(data_mod, "_encode_batch", fake_encode)

    # call 1: history=[turn0], target=turn1 -> embeds turn0, turn1
    build_context("sim", synthetic_dialogue[1], synthetic_dialogue[:1], k=1)
    # call 2: history=[turn0, turn1], target=turn2 -> turn0/turn1 already
    # cached, only turn2 needs embedding
    build_context("sim", synthetic_dialogue[2], synthetic_dialogue[:2], k=1)

    total_texts_embedded = sum(len(c) for c in calls)
    assert total_texts_embedded == 3  # turn0, turn1, turn2 -- each embedded once
