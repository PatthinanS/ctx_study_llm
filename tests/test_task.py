import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.run import (
    _record_succeeded,
    apply_task_namespacing,
    apply_task_override,
    load_config,
    load_done_ids,
    process_one,
)


def _base_cfg(tmp_path, extra=None):
    cfg = {
        "experiment_name": "task_test",
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


def test_load_config_without_task_defaults_to_both(tmp_path):
    cfg = load_config(str(_base_cfg(tmp_path)))
    assert cfg["task"] == "both"


@pytest.mark.parametrize("task", ["vad", "cat", "both"])
def test_load_config_with_explicit_task_is_read_through(tmp_path, task):
    cfg = load_config(str(_base_cfg(tmp_path, {"task": task})))
    assert cfg["task"] == task


def test_load_config_rejects_invalid_task(tmp_path):
    with pytest.raises(ValueError):
        load_config(str(_base_cfg(tmp_path, {"task": "bogus"})))


def test_apply_task_override_none_leaves_cfg_unchanged():
    cfg = {"task": "both"}
    assert apply_task_override(cfg, None)["task"] == "both"


def test_apply_task_override_sets_task():
    cfg = {"task": "both"}
    assert apply_task_override(cfg, "vad")["task"] == "vad"


def test_apply_task_namespacing_noop_for_both():
    cfg = {"task": "both", "experiment_name": "exp", "output_dir": "outputs"}
    result = apply_task_namespacing(cfg)
    assert result["experiment_name"] == "exp"
    assert result["output_dir"] == "outputs"


@pytest.mark.parametrize("task", ["vad", "cat"])
def test_apply_task_namespacing_suffixes_and_nests_dir(task):
    cfg = {"task": task, "experiment_name": "exp", "output_dir": "outputs"}
    result = apply_task_namespacing(cfg)
    assert result["experiment_name"] == f"exp_{task}"
    assert result["output_dir"] == str(Path("outputs") / task)


def _render_fn(row, history):
    return f'Utterance: "{row["text"]}"'


def test_process_one_vad_task_sends_vad_only_schema():
    row = {"utterance_id": "u1", "text": "hi", "emotion": "hap", "valence": 3.0, "arousal": 3.0, "dominance": 3.0}
    cfg = {"model": "fake-model", "condition": "C0"}
    response = {"message": {"content": json.dumps({"vad": {"v": 4.0, "a": 3.0, "d": 3.0}})}}
    mock_chat = MagicMock(return_value=response)
    with patch("ollama.chat", mock_chat):
        from src.prompts import RESPONSE_SCHEMA_VAD, SYSTEM_PROMPT_VAD

        record = process_one(
            row, [], cfg, _render_fn, {"temperature": 0, "seed": 42}, SYSTEM_PROMPT_VAD, RESPONSE_SCHEMA_VAD
        )
    assert mock_chat.call_args.kwargs["format"] == RESPONSE_SCHEMA_VAD
    assert record["pred_label"] is None
    assert record["pred_vad"] == {"v": 4.0, "a": 3.0, "d": 3.0}


def test_process_one_cat_task_sends_cat_only_schema():
    row = {"utterance_id": "u1", "text": "hi", "emotion": "hap", "valence": 3.0, "arousal": 3.0, "dominance": 3.0}
    cfg = {"model": "fake-model", "condition": "C0"}
    response = {"message": {"content": json.dumps({"label": "hap"})}}
    mock_chat = MagicMock(return_value=response)
    with patch("ollama.chat", mock_chat):
        from src.prompts import RESPONSE_SCHEMA_CAT, SYSTEM_PROMPT_CAT

        record = process_one(
            row, [], cfg, _render_fn, {"temperature": 0, "seed": 42}, SYSTEM_PROMPT_CAT, RESPONSE_SCHEMA_CAT
        )
    assert mock_chat.call_args.kwargs["format"] == RESPONSE_SCHEMA_CAT
    assert record["pred_label"] == "hap"
    assert record["pred_vad"] == {"v": None, "a": None, "d": None}


def test_record_succeeded_vad_task_requires_full_vad_triple():
    assert _record_succeeded({"pred_vad": {"v": 3.0, "a": 3.0, "d": 3.0}}, "vad") is True
    assert _record_succeeded({"pred_vad": {"v": 3.0, "a": None, "d": 3.0}}, "vad") is False
    assert _record_succeeded({"pred_vad": None}, "vad") is False


def test_record_succeeded_cat_task_requires_label():
    assert _record_succeeded({"pred_label": "hap"}, "cat") is True
    assert _record_succeeded({"pred_label": None}, "cat") is False


def test_load_done_ids_vad_task_treats_null_label_as_done(tmp_path):
    """Regression test: a "vad" run's pred_label is always None by design
    (no label was ever requested), so the task-aware check must not treat
    that as a failed/undone record the way the default "both" check would.
    """
    preds_path = tmp_path / "preds.jsonl"
    preds_path.write_text(
        json.dumps(
            {"utterance_id": "u1", "pred_label": None, "pred_vad": {"v": 3.0, "a": 3.0, "d": 3.0}}
        )
        + "\n"
        + json.dumps(
            {"utterance_id": "u2", "pred_label": None, "pred_vad": {"v": None, "a": None, "d": None}}
        )
        + "\n"
    )
    assert load_done_ids(preds_path, task="vad") == {"u1"}
    assert load_done_ids(preds_path, task="both") == set()
