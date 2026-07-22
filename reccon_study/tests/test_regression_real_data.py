"""Regression tests against the real IEMOCAP CSV + RECCON json. Skipped
automatically when either file is absent (mirrors tests/test_data.py's
Session5-split-parity skip pattern). The RECCON json isn't checked into this
repo -- place it at data/reccon/iemocap_test.json to run these.
"""
from pathlib import Path

import pytest

from reccon_study.align import build_alignment
from reccon_study.cli import ALL_SESSIONS
from reccon_study.eligibility import compute_funnel
from reccon_study.leakage import build_utterance_index
from src.data import load_iemocap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_CSV = REPO_ROOT / "data" / "iemocap" / "iemocap_merged_all.csv"
REAL_RECCON = REPO_ROOT / "data" / "reccon" / "iemocap_test.json"

pytestmark = pytest.mark.skipif(
    not (REAL_CSV.exists() and REAL_RECCON.exists()),
    reason="real IEMOCAP CSV and/or RECCON json not present in data/ "
    "(place RECCON json at data/reccon/iemocap_test.json)",
)


def test_real_alignment_16_dialogues_662_plus_matched():
    result, stats = build_alignment(str(REAL_RECCON), str(REAL_CSV))
    assert len(result) == 16
    assert stats["tot_matched"] >= 662


def _funnel_for_scope(sessions: list[str]) -> dict:
    result, _stats = build_alignment(str(REAL_RECCON), str(REAL_CSV))
    df = load_iemocap(str(REAL_CSV))
    utt_index = build_utterance_index(df, sessions=ALL_SESSIONS)
    targets = [
        u for dialog, record in result.items() if record["session"] in sessions
        for u in record["utterances"]
    ]
    return compute_funnel(
        targets,
        in_preds_fn=lambda t: True,
        pool_size_fn=lambda t: utt_index[t["utterance_id"]][2],
        k=4,
    )


def test_funnel_all_16_dialogues():
    funnel = _funnel_for_scope(ALL_SESSIONS)
    assert funnel["aligned"] == 662
    assert funnel["aligned"] - funnel["no_cause"] == 473
    assert funnel["no_in_pool_cause"] == 92
    assert funnel["forced_selection"] == 15
    assert funnel["scoreable"] == 366


def test_funnel_s4_s5():
    funnel = _funnel_for_scope(["Session4", "Session5"])
    assert funnel["aligned"] == 314
    assert funnel["aligned"] - funnel["no_cause"] == 236
    assert funnel["no_in_pool_cause"] == 40
    assert funnel["forced_selection"] == 2
    assert funnel["scoreable"] == 194


def test_funnel_s5_only():
    funnel = _funnel_for_scope(["Session5"])
    assert funnel["aligned"] == 124
    assert funnel["aligned"] - funnel["no_cause"] == 101
    assert funnel["no_in_pool_cause"] == 8
    assert funnel["forced_selection"] == 1
    assert funnel["scoreable"] == 92
