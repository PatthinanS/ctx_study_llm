import pytest

from reccon_study.align import align_one

SESSION = "Session1"
DIALOG = "Ses01F_impro01"


@pytest.fixture
def synthetic_csv_rows() -> list[dict]:
    """8-turn dialogue, full REQUIRED_COLS row-dicts, mirrors the style of
    tests/conftest.py's synthetic_dialogue fixture (repo-level). RECCON will
    drop positions 2 and 5 in `synthetic_reccon_turns` below."""
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
            "session": SESSION,
            "dialog": DIALOG,
            "utterance_id": f"{DIALOG}_{speakers[i]}{i:03d}",
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


@pytest.fixture
def synthetic_reccon_turns() -> list[dict]:
    """Parallel RECCON turn list: drops csv positions 2 and 5 entirely, and
    turn 3's `utterance` is a >=8-char fragment fully contained in csv
    position 3's text (not an exact match, and its containment ratio -- 20
    normalised chars in a 56-char string, ~0.36 -- is BELOW the plain 0.6
    guard) -- exercises the loosened length-only containment fallback.
    Turn 4's cause evidence resolves to turn 3 (a real in-list turn); turn
    5's references a turn number absent from this list (-> unresolved); turn
    6's is latent-only ("b", no integer entries)."""
    return [
        {"turn": 1, "utterance": "Hello how are you today", "emotion": "neu",
         "expanded emotion cause evidence": [], "type": []},
        {"turn": 2, "utterance": "I am fine thanks for asking", "emotion": "neu",
         "expanded emotion cause evidence": [], "type": []},
        {"turn": 3, "utterance": "beating around the bush", "emotion": "ang",
         "expanded emotion cause evidence": [], "type": ["no-context"]},
        {"turn": 4, "utterance": "What do you mean by that", "emotion": "sad",
         "expanded emotion cause evidence": [3], "type": ["inter-personal"]},
        {"turn": 5, "utterance": "Fine let us change the subject", "emotion": "neu",
         "expanded emotion cause evidence": [99], "type": ["hybrid"]},
        {"turn": 6, "utterance": "Sounds good to me", "emotion": "hap",
         "expanded emotion cause evidence": ["b"], "type": ["latent"]},
    ]


@pytest.fixture
def fake_dialogues(synthetic_csv_rows) -> dict[tuple[str, str], list[dict]]:
    """reconstruct_dialogues-shaped: {(session, dialog): [row, ...]}."""
    return {(SESSION, DIALOG): synthetic_csv_rows}


@pytest.fixture
def fake_alignment(synthetic_reccon_turns, synthetic_csv_rows) -> dict:
    """Small hand-computed-via-align_one alignment dict, matching the real
    reccon_ie_aligned.json schema for the synthetic dialogue above."""
    record, _unmatched = align_one(SESSION, synthetic_reccon_turns, synthetic_csv_rows)
    return {DIALOG: record}
