"""Data loading, splits, and context construction for the LLM prompting leg.

Data conventions mirror the sibling PLM repo (ctx_study) for CSV columns,
dialogue reconstruction, and leave-one-session-out splits, with one
intentional divergence: rows are NOT dropped for missing VAD at load time
(the PLM repo does `dropna` here). Doing so would break the split-parity
test (Session5 has 2195 raw rows; 25 have NaN VAD) and would silently
exclude valid context turns. NaN-gold-VAD rows are instead skipped only at
dimensional scoring time, in score.py.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

REQUIRED_COLS = {
    "session",
    "dialog",
    "utterance_id",
    "speaker",
    "start_time",
    "end_time",
    "text",
    "emotion",
    "valence",
    "arousal",
    "dominance",
}

LABELS = ["ang", "hap", "exc", "neu", "sad", "fru"]
EXCLUDED_EMOTIONS = {"xxx", "sur", "fea", "dis", "oth", ""}


def load_iemocap(csv_path: str) -> pd.DataFrame:
    """Load the merged IEMOCAP CSV as-is.

    No dropna, no session dtype coercion: session values stay plain strings
    ("Session1".."Session5") throughout, compared via direct equality/.isin()
    everywhere downstream.
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"IEMOCAP CSV missing columns: {sorted(missing)}")
    df["session"] = df["session"].astype(str)
    return df


def is_categorical_usable(row: dict | pd.Series) -> bool:
    """True if row['emotion'] is a valid categorical scoring target.

    NaN/missing emotion and rows in EXCLUDED_EMOTIONS return False. Such
    rows are still valid dimensional (VAD) scoring targets and are always
    usable as context turns -- this predicate only governs whether a row
    counts as a categorical gold label.
    """
    emotion = row.get("emotion") if isinstance(row, dict) else row["emotion"]
    if pd.isna(emotion):
        return False
    return str(emotion) in LABELS


def get_splits(cfg: dict) -> tuple[list[str], str, str]:
    """Read (train_sessions, val_session, test_session) directly from config.

    Sessions are plain strings (e.g. "Session5"), authoritative from config
    -- no inference from the dataframe. Raises ValueError on overlap.
    """
    splits = cfg["splits"]
    train = list(splits["train_sessions"])
    val = splits["val_session"]
    test = splits["test_session"]
    overlap = set(train) & {val, test}
    if overlap:
        raise ValueError(f"Split overlap between train/val/test: {overlap}")
    return train, val, test


def reconstruct_dialogues(
    df: pd.DataFrame, sessions: list[str]
) -> dict[tuple[str, Any], list[dict]]:
    """Group by (session, dialog), sort each group by start_time.

    Returns {(session, dialog): [row_dict, ...]} in chronological order.
    """
    subset = df[df["session"].isin(sessions)]
    out: dict[tuple[str, Any], list[dict]] = {}
    for key, grp in subset.groupby(["session", "dialog"], sort=False):
        out[key] = grp.sort_values("start_time").to_dict("records")
    return out


def iter_eval_rows(df: pd.DataFrame, session: str):
    """Yield (row, history) for every utterance in the given session.

    Iterates dialogues in groupby order, and within each dialogue in
    chronological order. `history` is the list of prior-turn row dicts in
    the same dialogue (chronological), regardless of categorical
    usability -- all prior text is usable as context.
    """
    dialogues = reconstruct_dialogues(df, [session])
    for _key, rows in dialogues.items():
        for idx, row in enumerate(rows):
            yield row, rows[:idx]


# ---------------------------------------------------------------------------
# Context strategy registry (extensibility seam for C2 "retrieval" later)
# ---------------------------------------------------------------------------


def _strategy_none(utterance: dict, history: list[dict], k: int, **kwargs) -> list[dict]:
    return []


def _strategy_window(utterance: dict, history: list[dict], k: int, **kwargs) -> list[dict]:
    """Last k prior turns, chronological order. [] if k<=0 or no history."""
    if k <= 0 or not history:
        return []
    return history[-k:]


STRATEGY_REGISTRY: dict[str, Callable[..., list[dict]]] = {
    "none": _strategy_none,
    "window": _strategy_window,
    # "retrieval": _strategy_retrieval,  # C2 plugs in here, same signature
}


def build_context(
    strategy: str, utterance: dict, history: list[dict], k: int, **kwargs
) -> list[dict]:
    """Dispatch to a named context strategy.

    Returns a list of turn dicts (a subset of history, chronological order)
    -- not a pre-formatted string. Formatting into "SPEAKER: text" lines
    happens in prompts.py, keeping strategy functions reusable unchanged
    when a new prompt template is introduced.
    """
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown context strategy '{strategy}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[strategy](utterance, history, k, **kwargs)
