import pytest


@pytest.fixture
def synthetic_dialogue():
    """A 6-turn dialogue in the same (session, dialog), chronological order.

    Turn index 3 has emotion="xxx" (excluded from categorical scoring, but
    still a valid context turn), to exercise that distinction in tests.
    """
    speakers = ["A", "B", "A", "B", "A", "B"]
    emotions = ["neu", "hap", "neu", "xxx", "sad", "fru"]
    turns = []
    for i in range(6):
        turns.append(
            {
                "utterance_id": f"Ses01F_impro01_{i:03d}",
                "session": "Session1",
                "dialog": "impro01",
                "speaker": speakers[i],
                "start_time": float(i),
                "end_time": float(i) + 0.5,
                "text": f"utterance number {i}",
                "emotion": emotions[i],
                "valence": 3.0,
                "arousal": 3.0,
                "dominance": 3.0,
            }
        )
    return turns
