import json
from unittest.mock import MagicMock, patch

import pytest

from src.prompts import SYSTEM_PROMPT, build_few_shot_block
from src.run import load_config, process_one


def _base_cfg(tmp_path, extra=None):
    cfg = {
        "experiment_name": "fewshot_test",
        "model": "fake-model",
        "csv_path": "data/iemocap/iemocap_merged_all.csv",
        "splits": {
            "test_session": "Session5",
            "val_session": "Session4",
            "train_sessions": ["Session1", "Session2", "Session3"],
        },
        "condition": "C0",
        "context": {"strategy": "none", "k": 0, "strategy_kwargs": {}},
        "labels": ["ang", "hap", "exc", "neu", "sad", "fru"],
        "output_dir": str(tmp_path),
        "temperature": 0,
        "seed": 42,
    }
    if extra:
        cfg.update(extra)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path


def test_load_config_without_few_shot_defaults_to_none(tmp_path):
    cfg = load_config(str(_base_cfg(tmp_path)))
    assert cfg["few_shot"] is None


def test_load_config_with_few_shot_is_read_through(tmp_path):
    cfg = load_config(str(_base_cfg(tmp_path, {"few_shot": {"n": 4}})))
    assert cfg["few_shot"] == {"n": 4}


def test_load_config_rejects_few_shot_without_n(tmp_path):
    with pytest.raises(ValueError):
        load_config(str(_base_cfg(tmp_path, {"few_shot": {}})))


def test_load_config_rejects_non_positive_n(tmp_path):
    with pytest.raises(ValueError):
        load_config(str(_base_cfg(tmp_path, {"few_shot": {"n": 0}})))


def _render_fn(row, history):
    return f'Utterance: "{row["text"]}"'


def _response():
    return {"message": {"content": json.dumps({"label": "hap", "vad": {"v": 4.0, "a": 3.0, "d": 3.0}})}}


def test_process_one_default_system_prompt_is_zero_shot():
    row = {"utterance_id": "u1", "text": "hi", "emotion": "hap", "valence": 3.0, "arousal": 3.0, "dominance": 3.0}
    cfg = {"model": "fake-model", "condition": "C0"}
    mock_chat = MagicMock(return_value=_response())
    with patch("ollama.chat", mock_chat):
        process_one(row, [], cfg, _render_fn, {"temperature": 0, "seed": 42})
    sent_system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert sent_system == SYSTEM_PROMPT
    assert "EXAMPLES" not in sent_system


def test_process_one_with_few_shot_system_prompt_includes_block():
    row = {"utterance_id": "u1", "text": "hi", "emotion": "hap", "valence": 3.0, "arousal": 3.0, "dominance": 3.0}
    cfg = {"model": "fake-model", "condition": "C0"}
    block = build_few_shot_block(
        [{"text": "example text", "label": "sad", "vad": {"v": 1.5, "a": 2.0, "d": 2.5}}]
    )
    few_shot_system_prompt = f"{SYSTEM_PROMPT}\n\n{block}"

    mock_chat = MagicMock(return_value=_response())
    with patch("ollama.chat", mock_chat):
        process_one(row, [], cfg, _render_fn, {"temperature": 0, "seed": 42}, few_shot_system_prompt)
    sent_system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert sent_system != SYSTEM_PROMPT
    assert "EXAMPLES" in sent_system
    assert "example text" in sent_system
