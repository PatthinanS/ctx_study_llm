import json
from unittest.mock import MagicMock, patch

from src.run import process_one_c2c, select_stage1


def _cfg(model="fake-model"):
    return {"model": model, "condition": "C2", "temperature": 0, "seed": 42}


def _options():
    return {"temperature": 0, "seed": 42}


def _selection_response(indices):
    return {"message": {"content": json.dumps({"selected": indices})}}


def test_stage1_accepts_valid_selection(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]  # pool_size=5, k=3 -> n_sel=3
    mock_chat = MagicMock(return_value=_selection_response([0, 1, 2]))
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["selected_indices"] == [0, 1, 2]
    assert result["fallback"] is False
    assert result["stage1_skipped"] is False
    assert mock_chat.call_count == 1


def test_stage1_out_of_range_falls_back_to_recency(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    mock_chat = MagicMock(return_value=_selection_response([0, 1, 99]))
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["fallback"] is True
    assert result["selected_indices"] == [2, 3, 4]  # recency: last n_sel=3 of pool_size=5
    assert mock_chat.call_count == 2


def test_stage1_duplicate_indices_falls_back(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    mock_chat = MagicMock(return_value=_selection_response([0, 0, 1]))
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["fallback"] is True
    assert mock_chat.call_count == 2


def test_stage1_wrong_length_falls_back(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    mock_chat = MagicMock(return_value=_selection_response([0, 1]))
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["fallback"] is True
    assert mock_chat.call_count == 2


def test_stage1_non_json_falls_back(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    mock_chat = MagicMock(return_value={"message": {"content": "not json"}})
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["fallback"] is True
    assert mock_chat.call_count == 2


def test_stage1_retries_once_then_succeeds(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    mock_chat = MagicMock(
        side_effect=[
            {"message": {"content": "not json"}},
            _selection_response([0, 1, 2]),
        ]
    )
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["fallback"] is False
    assert result["selected_indices"] == [0, 1, 2]
    assert mock_chat.call_count == 2


def test_stage1_skipped_when_pool_le_n_sel(synthetic_dialogue):
    target = synthetic_dialogue[2]
    history = synthetic_dialogue[:2]  # pool_size=2, k=3 -> n_sel=2, pool<=n_sel
    mock_chat = MagicMock()
    with patch("ollama.chat", mock_chat):
        result = select_stage1(target, history, _cfg(), k=3, kwargs={}, options=_options())
    assert result["stage1_skipped"] is True
    assert result["selected_indices"] == [0, 1]
    assert mock_chat.call_count == 0


def test_process_one_c2c_end_to_end(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    stage1_resp = _selection_response([0, 1, 2])
    stage2_resp = {
        "message": {"content": json.dumps({"label": "hap", "vad": {"v": 4.0, "a": 3.0, "d": 3.0}})}
    }
    mock_chat = MagicMock(side_effect=[stage1_resp, stage2_resp])
    with patch("ollama.chat", mock_chat):
        record = process_one_c2c(target, history, _cfg(), k=3, kwargs={}, options=_options())

    for key in (
        "utterance_id",
        "condition",
        "model",
        "gold_label",
        "gold_vad",
        "pred_label",
        "pred_vad",
        "latency_ms",
        "raw_response",
        "selected_indices",
        "pool_size",
        "fallback",
        "stage1_skipped",
        "stage1_latency_ms",
        "stage2_latency_ms",
        "stage1_raw_response",
    ):
        assert key in record

    assert record["selected_indices"] == [0, 1, 2]
    assert record["pool_size"] == 5
    assert record["pred_label"] == "hap"
    assert record["fallback"] is False
    assert record["stage1_skipped"] is False
