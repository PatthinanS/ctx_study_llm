"""System/user prompt templates and the Ollama response schema.

SYSTEM_PROMPT is a single constant, identical for C0 and C1, logged into
run_meta.json for every run. condition ("C0"/"C1" -> which template) is
decoupled from context.strategy ("none"/"window"/"retrieval" -> which turns
get selected) so a future C2 (retrieval) condition can reuse the C1 template
unchanged -- only src/data.py's STRATEGY_REGISTRY needs a new entry.
"""
from __future__ import annotations

import json

from src.data import LABELS

SYSTEM_PROMPT = """You are an expert annotator for emotion recognition in conversation (IEMOCAP).

TASK: Given a single TARGET utterance from a dyadic conversation (optionally with preceding context), predict:
1. A categorical emotion label for the TARGET utterance only.
2. Continuous Valence/Arousal/Dominance (VAD) ratings for the TARGET utterance only.

CATEGORICAL LABELS (choose exactly one):
- ang: angry
- hap: happy
- exc: excited
- neu: neutral
- sad: sad
- fru: frustrated

VAD DIMENSIONS (each on a continuous 1.0-5.0 scale, one decimal place allowed):
- v (valence): 1.0 = very negative, 5.0 = very positive
- a (arousal): 1.0 = very calm, 5.0 = very activated
- d (dominance): 1.0 = very submissive, 5.0 = very dominant

OUTPUT FORMAT: Respond with JSON only, matching the required schema exactly. No prose, no explanation, no markdown fences."""


def build_user_prompt_c0(text: str) -> str:
    return f'Utterance: "{text}"'


def build_user_prompt_c1(context_turns: list[dict], speaker: str, text: str) -> str:
    """Render the C1 (windowed-context) user prompt.

    context_turns: list of {"speaker": ..., "text": ...} dicts, chronological.
    If empty (first utterance of a dialogue), degrades to the C0 template.
    """
    if not context_turns:
        return build_user_prompt_c0(text)
    lines = "\n".join(f"{t['speaker']}: {t['text']}" for t in context_turns)
    return (
        f"Context:\n{lines}\n\n"
        f'TARGET {speaker}: "{text}"\n\n'
        "Classify and rate only the TARGET utterance above; the Context lines are for reference only."
    )


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "vad": {
            "type": "object",
            "properties": {
                "v": {"type": "number"},
                "a": {"type": "number"},
                "d": {"type": "number"},
            },
            "required": ["v", "a", "d"],
        },
    },
    "required": ["label", "vad"],
}


def build_few_shot_block(examples: list[dict]) -> str:
    """Render fixed text -> {label, vad} demonstrations, matching
    RESPONSE_SCHEMA's exact output shape, so the model sees the required
    format and the 1.0-5.0 VAD scale together.

    examples: list of {"text": str, "label": str, "vad": {"v","a","d": float}}.
    """
    blocks = []
    for i, ex in enumerate(examples, 1):
        output = json.dumps({"label": ex["label"], "vad": ex["vad"]})
        blocks.append(f'Example {i}:\nUtterance: "{ex["text"]}"\nOutput: {output}')
    return (
        "EXAMPLES (showing the required output format and the 1.0-5.0 VAD scale; "
        "illustrative only, unrelated to the conversation below):\n\n" + "\n\n".join(blocks)
    )


def build_system_prompt(few_shot_block: str | None = None) -> str:
    """SYSTEM_PROMPT, optionally with a few-shot examples block appended.

    Returns SYSTEM_PROMPT unchanged (same object) when few_shot_block is
    falsy, so the zero-shot path stays byte-for-byte identical.
    """
    if not few_shot_block:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n{few_shot_block}"


SELECTION_SYSTEM_PROMPT = """You are selecting which prior conversation turns to give to a downstream emotion-recognition annotator as context for a single TARGET utterance.

TASK: You will see all prior turns of the dialogue, each numbered with its pool index in brackets (e.g. "[0] SPEAKER: text"), followed by the TARGET utterance. Select exactly N of the numbered prior turns that are most useful for inferring the TARGET speaker's emotional state in the TARGET utterance.

Judge usefulness by EMOTIONAL RELEVANCE ONLY -- e.g. turns that reveal a mood shift, an emotional trigger, or the speaker's affective trajectory leading into the TARGET line. Do NOT select turns merely because they discuss the same topic or share vocabulary with the TARGET utterance; topical similarity is not the criterion.

OUTPUT FORMAT: Respond with JSON only, matching the required schema exactly (a "selected" array of exactly N distinct pool indices, each in range). No prose, no explanation, no markdown fences."""


def build_user_prompt_selection(history: list[dict], speaker: str, text: str, n_sel: int) -> str:
    """Numbered-pool selection prompt for C2c stage 1.

    history: list of {"speaker": ..., "text": ...} dicts, chronological --
    the same pool a context strategy would receive. Assumes history is
    non-empty (caller only invokes this when pool_size > n_sel, which
    implies pool_size >= 1).
    """
    lines = "\n".join(f"[{i}] {t['speaker']}: {t['text']}" for i, t in enumerate(history))
    return (
        f"Prior turns (pool indices in brackets):\n{lines}\n\n"
        f'TARGET {speaker}: "{text}"\n\n'
        f"Select exactly {n_sel} of the numbered prior turn indices above that are most useful for "
        f"inferring {speaker}'s emotional state in the TARGET utterance. Judge by emotional relevance, "
        "not topical similarity."
    )


def build_selection_schema(n_sel: int) -> dict:
    """Dynamic JSON schema, minItems==maxItems==n_sel per call (n_sel varies
    per utterance since it's min(cfg k, pool_size))."""
    return {
        "type": "object",
        "properties": {
            "selected": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": n_sel,
                "maxItems": n_sel,
            }
        },
        "required": ["selected"],
    }
