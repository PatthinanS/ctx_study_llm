import json

import pandas as pd
import pytest

from reccon_study.leakage import (
    build_alignment_lookup,
    build_utterance_index,
    classify_selection,
    derive_recency_selection,
    get_selection,
    paired_bootstrap_diff,
    run_leakage,
)


def test_build_utterance_index_matches_pool_size(fake_dialogues):
    rows = fake_dialogues[("Session1", "Ses01F_impro01")]
    df = pd.DataFrame(rows)
    idx = build_utterance_index(df, sessions=["Session1"])
    assert idx[rows[0]["utterance_id"]] == ("Session1", "Ses01F_impro01", 0)
    assert idx[rows[3]["utterance_id"]] == ("Session1", "Ses01F_impro01", 3)
    assert idx[rows[7]["utterance_id"]] == ("Session1", "Ses01F_impro01", 7)


def test_derive_recency_selection_first_utterance_empty_pool():
    assert derive_recency_selection(pool_size=0, k=4) == []


def test_derive_recency_selection_pool_smaller_than_k_selects_all():
    assert derive_recency_selection(pool_size=3, k=4) == [0, 1, 2]


def test_derive_recency_selection_last_k():
    assert derive_recency_selection(pool_size=10, k=4) == [6, 7, 8, 9]


def test_get_selection_c1_record_has_no_selected_indices():
    rec = {"utterance_id": "u1", "condition": "C1"}  # no selected_indices/pool_size
    assert get_selection(rec, pool_size=10, k=4) == [6, 7, 8, 9]


def test_get_selection_c2_record_returns_stored_indices():
    rec = {"utterance_id": "u1", "selected_indices": [1, 3], "pool_size": 5}
    assert get_selection(rec, pool_size=5, k=2) == [1, 3]


def test_get_selection_c2_record_pool_size_mismatch_raises():
    rec = {"utterance_id": "u1", "selected_indices": [1, 3], "pool_size": 5}
    with pytest.raises(AssertionError):
        get_selection(rec, pool_size=999, k=2)


def test_classify_selection_gold_cause():
    assert classify_selection("u1", cause_uids={"u1", "u2"}, visible_uids={"u1", "u2", "u3"}) == "gold_cause"


def test_classify_selection_visible_noncause():
    assert classify_selection("u3", cause_uids={"u1"}, visible_uids={"u1", "u3"}) == "visible_noncause"


def test_classify_selection_invisible():
    assert classify_selection("u9", cause_uids={"u1"}, visible_uids={"u1", "u3"}) == "invisible"


def test_build_alignment_lookup(fake_alignment):
    target_by_uid, visible_uids_by_dialog = build_alignment_lookup(fake_alignment)
    dialog = "Ses01F_impro01"
    assert visible_uids_by_dialog[dialog] == {
        u["utterance_id"] for u in fake_alignment[dialog]["utterances"]
    }
    turn4_uid = next(
        u["utterance_id"] for u in fake_alignment[dialog]["utterances"] if u["reccon_turn"] == 4
    )
    assert target_by_uid[turn4_uid]["cause_csv_pos"] == [3]


def test_paired_bootstrap_diff_deterministic_by_seed():
    observed = [0.3, 0.5, 0.1, 0.4, 0.2]
    baseline = [0.4, 0.4, 0.2, 0.3, 0.3]
    r1 = paired_bootstrap_diff(observed, baseline, n_resamples=500, seed=7)
    r2 = paired_bootstrap_diff(observed, baseline, n_resamples=500, seed=7)
    assert r1 == r2


def test_paired_bootstrap_diff_different_seed_can_differ():
    observed = [0.3, 0.5, 0.1, 0.4, 0.2]
    baseline = [0.4, 0.4, 0.2, 0.3, 0.3]
    r1 = paired_bootstrap_diff(observed, baseline, n_resamples=500, seed=7)
    r2 = paired_bootstrap_diff(observed, baseline, n_resamples=500, seed=8)
    assert r1["ci_lo"] != r2["ci_lo"] or r1["ci_hi"] != r2["ci_hi"]


def test_paired_bootstrap_diff_point_is_mean_of_differences():
    observed = [0.5, 0.5, 0.5]
    baseline = [0.2, 0.2, 0.2]
    r = paired_bootstrap_diff(observed, baseline, n_resamples=100, seed=1)
    assert r["diff"] == pytest.approx(0.3)


def test_run_leakage_excludes_targets_not_in_preds(tmp_path, fake_alignment):
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    dialog = "Ses01F_impro01"
    turn4 = next(u for u in fake_alignment[dialog]["utterances"] if u["reccon_turn"] == 4)
    # only turn4's utterance_id is in preds -- every other aligned target should
    # land in "not_in_preds" and be excluded from all leakage stats.
    with open(run_dir / "preds.jsonl", "w") as f:
        f.write(json.dumps({
            "utterance_id": turn4["utterance_id"], "selected_indices": [3], "pool_size": 4,
        }) + "\n")

    csv_path = tmp_path / "iemocap.csv"
    rows = _rows_from_fixture()
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    report = run_leakage([run_dir], fake_alignment, str(csv_path), sessions=["Session1"],
                          seed=1, n_resamples=50)
    data = report[run_dir.name]
    assert data["funnel"]["aligned"] == 6
    assert data["funnel"]["not_in_preds"] == 5


def _rows_from_fixture():
    texts = [
        "Hello how are you today",
        "I am fine thanks for asking",
        "That is good to hear friend",
        "Stop beating around the bush and just tell me exactly what happened",
        "What do you mean by that",
        "Nothing forget I said it",
        "Fine let us change the subject",
        "Sounds good to me",
    ]
    speakers = ["F", "M", "F", "M", "F", "M", "F", "M"]
    emotions = ["neu", "neu", "neu", "ang", "sad", "neu", "neu", "hap"]
    rows = []
    for i in range(8):
        rows.append({
            "session": "Session1",
            "dialog": "Ses01F_impro01",
            "utterance_id": f"Ses01F_impro01_{speakers[i]}{i:03d}",
            "speaker": speakers[i],
            "start_time": float(i),
            "end_time": float(i) + 0.9,
            "text": texts[i],
            "emotion": emotions[i],
            "valence": 3.0,
            "arousal": 3.0,
            "dominance": 3.0,
        })
    return rows
