"""System/user prompt templates and the Ollama response schema.

SYSTEM_PROMPT is a single constant, identical for C0 and C1, logged into
run_meta.json for every run. condition ("C0"/"C1" -> which template) is
decoupled from context.strategy ("none"/"window"/"retrieval" -> which turns
get selected) so a future C2 (retrieval) condition can reuse the C1 template
unchanged -- only src/data.py's STRATEGY_REGISTRY needs a new entry.
"""
from __future__ import annotations

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
