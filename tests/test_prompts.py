from src.data import LABELS, build_context
from src.prompts import (
    RESPONSE_SCHEMA,
    build_user_prompt_c0,
    build_user_prompt_c1,
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
