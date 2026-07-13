import json
from unittest.mock import MagicMock, patch

from src.run import (
    call_ollama_with_retry,
    load_done_ids,
    validate_and_clamp,
)


def test_load_done_ids_skips_completed(tmp_path):
    preds_path = tmp_path / "preds.jsonl"
    preds_path.write_text(
        json.dumps({"utterance_id": "u1"}) + "\n" + json.dumps({"utterance_id": "u2"}) + "\n"
    )
    assert load_done_ids(preds_path) == {"u1", "u2"}


def test_load_done_ids_missing_file(tmp_path):
    assert load_done_ids(tmp_path / "does_not_exist.jsonl") == set()


def test_load_done_ids_tolerates_corrupt_trailing_line(tmp_path):
    preds_path = tmp_path / "preds.jsonl"
    preds_path.write_text(
        json.dumps({"utterance_id": "u1"}) + "\n" + '{"utterance_id": "u2", "incomple'
    )
    assert load_done_ids(preds_path) == {"u1"}


def test_validate_and_clamp_out_of_range():
    parsed = {"label": "hap", "vad": {"v": 7.2, "a": -1, "d": 3.0}}
    label, vad = validate_and_clamp(parsed)
    assert label == "hap"
    assert vad == {"v": 5.0, "a": 1.0, "d": 3.0}


def test_validate_and_clamp_invalid_label():
    parsed = {"label": "bogus", "vad": {"v": 3.0, "a": 3.0, "d": 3.0}}
    label, vad = validate_and_clamp(parsed)
    assert label is None
    assert vad == {"v": 3.0, "a": 3.0, "d": 3.0}


def test_validate_and_clamp_missing_vad_fields():
    parsed = {"label": "neu", "vad": {"v": 3.0}}
    label, vad = validate_and_clamp(parsed)
    assert label == "neu"
    assert vad == {"v": 3.0, "a": None, "d": None}


def test_call_ollama_with_retry_succeeds_on_second_attempt():
    good_response = {
        "message": {"content": json.dumps({"label": "hap", "vad": {"v": 4.0, "a": 3.0, "d": 3.0}})}
    }
    mock_chat = MagicMock(side_effect=[RuntimeError("boom"), good_response])
    with patch("ollama.chat", mock_chat):
        label, vad, latency_ms, raw = call_ollama_with_retry(
            "fake-model", "sys", "user", {"temperature": 0, "seed": 42}, {}
        )
    assert label == "hap"
    assert vad == {"v": 4.0, "a": 3.0, "d": 3.0}
    assert mock_chat.call_count == 2


def test_call_ollama_with_retry_fails_gracefully_after_two_attempts():
    mock_chat = MagicMock(side_effect=[RuntimeError("boom"), RuntimeError("boom again")])
    with patch("ollama.chat", mock_chat):
        label, vad, latency_ms, raw = call_ollama_with_retry(
            "fake-model", "sys", "user", {"temperature": 0, "seed": 42}, {}
        )
    assert label is None
    assert vad == {"v": None, "a": None, "d": None}
    assert mock_chat.call_count == 2
