import pytest

from src.data import LABELS, build_context
from src.prompts import (
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_CAT,
    RESPONSE_SCHEMA_VAD,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_CAT,
    SYSTEM_PROMPT_VAD,
    build_few_shot_block,
    build_selection_schema,
    build_system_prompt,
    build_user_prompt_c0,
    build_user_prompt_c1,
    build_user_prompt_selection,
)


def test_c0_prompt_format():
    assert build_user_prompt_c0("hello there") == 'Utterance: "hello there"'


def test_c1_prompt_contains_target_marker_and_last_3_turns(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    turns = build_context("window", target, history, k=3)
    prompt = build_user_prompt_c1(turns, target["speaker"], target["text"])

    assert f'TARGET {target["speaker"]}: "{target["text"]}"' in prompt
    for turn in synthetic_dialogue[2:5]:
        assert f'{turn["speaker"]}: {turn["text"]}' in prompt
    for turn in synthetic_dialogue[:2]:
        assert turn["text"] not in prompt


def test_c1_empty_context_degrades_to_c0(synthetic_dialogue):
    target = synthetic_dialogue[0]
    prompt = build_user_prompt_c1([], target["speaker"], target["text"])
    assert prompt == build_user_prompt_c0(target["text"])
    assert "Context:" not in prompt


def test_response_schema_has_required_fields():
    assert RESPONSE_SCHEMA["properties"]["label"]["enum"] == LABELS
    assert set(RESPONSE_SCHEMA["properties"]["vad"]["properties"]) == {"v", "a", "d"}
    assert RESPONSE_SCHEMA["required"] == ["label", "vad"]


def test_selection_prompt_numbers_pool_and_marks_target(synthetic_dialogue):
    target = synthetic_dialogue[5]
    history = synthetic_dialogue[:5]
    prompt = build_user_prompt_selection(history, target["speaker"], target["text"], n_sel=3)

    for i, turn in enumerate(history):
        assert f'[{i}] {turn["speaker"]}: {turn["text"]}' in prompt
    assert f'TARGET {target["speaker"]}: "{target["text"]}"' in prompt


def test_selection_schema_min_max_items_match_n_sel():
    schema = build_selection_schema(4)
    assert schema["properties"]["selected"]["minItems"] == 4
    assert schema["properties"]["selected"]["maxItems"] == 4
    assert schema["required"] == ["selected"]


def _few_shot_examples():
    return [
        {"text": "I can't believe you did that.", "label": "ang", "vad": {"v": 1.5, "a": 4.5, "d": 4.0}},
        {"text": "That's wonderful news!", "label": "hap", "vad": {"v": 4.8, "a": 3.5, "d": 3.2}},
    ]


def test_build_few_shot_block_contains_examples_in_schema_format():
    block = build_few_shot_block(_few_shot_examples())
    assert "Example 1:" in block
    assert "Example 2:" in block
    assert 'Utterance: "I can\'t believe you did that."' in block
    assert '"label": "ang"' in block
    assert '"v": 1.5' in block
    assert '"a": 4.5' in block
    assert '"d": 4.0' in block


def test_build_system_prompt_none_is_identical_to_system_prompt():
    assert build_system_prompt(None) is SYSTEM_PROMPT
    assert build_system_prompt("") is SYSTEM_PROMPT


def test_build_system_prompt_with_block_contains_both():
    block = build_few_shot_block(_few_shot_examples())
    prompt = build_system_prompt(block)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert block in prompt


def test_response_schema_vad_requires_only_vad():
    assert set(RESPONSE_SCHEMA_VAD["properties"]) == {"vad"}
    assert set(RESPONSE_SCHEMA_VAD["properties"]["vad"]["properties"]) == {"v", "a", "d"}
    assert RESPONSE_SCHEMA_VAD["required"] == ["vad"]


def test_response_schema_cat_requires_only_label():
    assert set(RESPONSE_SCHEMA_CAT["properties"]) == {"label"}
    assert RESPONSE_SCHEMA_CAT["properties"]["label"]["enum"] == LABELS
    assert RESPONSE_SCHEMA_CAT["required"] == ["label"]


def test_system_prompt_vad_has_no_categorical_content():
    assert "CATEGORICAL LABELS" not in SYSTEM_PROMPT_VAD
    assert "ang: angry" not in SYSTEM_PROMPT_VAD
    assert "VAD DIMENSIONS" in SYSTEM_PROMPT_VAD


def test_system_prompt_vad_defines_each_dimension():
    for term in ("pleasantness", "activation", "control"):
        assert term in SYSTEM_PROMPT_VAD


def test_system_prompt_cat_has_no_vad_content():
    assert "VAD DIMENSIONS" not in SYSTEM_PROMPT_CAT
    assert "valence" not in SYSTEM_PROMPT_CAT.lower()
    assert "CATEGORICAL LABELS" in SYSTEM_PROMPT_CAT


@pytest.mark.parametrize(
    "task,expected",
    [("both", SYSTEM_PROMPT), ("vad", SYSTEM_PROMPT_VAD), ("cat", SYSTEM_PROMPT_CAT)],
)
def test_build_system_prompt_identity_per_task(task, expected):
    assert build_system_prompt(None, task) is expected
    assert build_system_prompt("", task) is expected


def test_build_few_shot_block_vad_task_omits_label():
    block = build_few_shot_block(_few_shot_examples(), task="vad")
    assert '"label"' not in block
    assert '"v": 1.5' in block


def test_build_few_shot_block_cat_task_omits_vad():
    block = build_few_shot_block(_few_shot_examples(), task="cat")
    assert '"vad"' not in block
    assert '"label": "ang"' in block
