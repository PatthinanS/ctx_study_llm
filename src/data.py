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

import hashlib
import random
import threading
from typing import Any, Callable

import numpy as np
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


def _stable_seed(seed: int, utterance_id: str) -> int:
    """Deterministic per-utterance seed via hashlib (not Python's salted hash())."""
    digest = hashlib.sha256(f"{seed}:{utterance_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _strategy_random(utterance: dict, history: list[dict], k: int, **kwargs) -> list[dict]:
    """Uniform sample of n_sel=min(k,|history|) prior turns, chronological order.

    Deterministic per (seed, utterance_id) via a local random.Random instance
    (never the global random module state) -- this makes the result depend
    only on (seed, utterance_id, history), independent of call order, which
    matters since main()'s ThreadPoolExecutor may interleave calls for
    different utterances arbitrarily.
    """
    n_sel = min(k, len(history))
    if n_sel <= 0 or not history:
        return []
    seed = kwargs.get("seed")
    if seed is None:
        raise ValueError("_strategy_random requires 'seed' in kwargs")
    rng = random.Random(_stable_seed(seed, utterance["utterance_id"]))
    indices = sorted(rng.sample(range(len(history)), n_sel))
    return [history[i] for i in indices]


_EMBED_LOCK = threading.Lock()
_EMBED_MODELS: dict[str, tuple] = {}
_EMBED_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_or_load_model(encoder_name: str):
    if encoder_name not in _EMBED_MODELS:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        device = _pick_device()
        model = AutoModel.from_pretrained(encoder_name).to(device).eval()
        _EMBED_MODELS[encoder_name] = (tokenizer, model, device)
    return _EMBED_MODELS[encoder_name]


def _encode_batch(texts: list[str], encoder_name: str) -> np.ndarray:
    """Mean-pool last hidden state over the attention mask, max_length=128.

    Mirrors the sibling PLM repo's _embed_dialogue exactly (same pooling
    formula, same "{speaker}: {text}" input format) for consistency between
    the two repos' retrieval logic. Sole seam touching torch/transformers --
    tests monkeypatch this function directly to inject fixed vectors.
    """
    import torch

    tokenizer, model, device = _get_or_load_model(encoder_name)
    enc = tokenizer(
        texts, max_length=128, truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled.cpu().numpy()


def _embed_dialogue_cached(
    session: str, dialog: Any, turns: list[dict], encoder_name: str
) -> dict[str, np.ndarray]:
    """Grow-only cache of turn embeddings, keyed by (session, dialog).

    "Embed each dialogue once" is reinterpreted as "never re-embed a turn
    whose text was already embedded": a strategy call only ever sees prior
    turns plus the current utterance, not the full future dialogue, so a
    true single upfront batch of the whole dialogue isn't available at call
    time. Lock is held for the whole cache-miss batch-encode call (not just
    the dict mutation) -- simple and correct under concurrent
    ThreadPoolExecutor workers; acceptable since embedding a handful of short
    texts is fast relative to the Ollama round-trip that dominates wall-clock
    time per utterance, and the Ollama call itself happens outside this lock.
    """
    key = (session, dialog)
    with _EMBED_LOCK:
        cache = _EMBED_CACHE.setdefault(key, {})
        missing = [t for t in turns if t["utterance_id"] not in cache]
        if missing:
            texts = [f"{t['speaker']}: {t['text']}" for t in missing]
            vecs = _encode_batch(texts, encoder_name)
            for t, v in zip(missing, vecs):
                cache[t["utterance_id"]] = v
        return cache


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _strategy_sim(utterance: dict, history: list[dict], k: int, **kwargs) -> list[dict]:
    """Top-n_sel prior turns by cosine similarity to the target, reordered
    chronologically before rendering (per the C2 shared-behavior contract)."""
    n_sel = min(k, len(history))
    if n_sel <= 0 or not history:
        return []
    encoder_name = kwargs.get("encoder", "xlm-roberta-base")
    emb_map = _embed_dialogue_cached(
        utterance["session"], utterance["dialog"], history + [utterance], encoder_name
    )
    target_vec = emb_map[utterance["utterance_id"]]
    sims = [(i, _cosine(target_vec, emb_map[h["utterance_id"]])) for i, h in enumerate(history)]
    sims.sort(key=lambda p: p[1], reverse=True)
    top_indices = sorted(i for i, _ in sims[:n_sel])
    return [history[i] for i in top_indices]


def select_few_shot_examples(
    df: pd.DataFrame, train_sessions: list[str], n: int, seed: int
) -> list[dict]:
    """Pick n fixed demonstration rows from train_sessions only.

    Spread across the dominance range (sort the eligible pool by dominance,
    split into n equal-sized buckets, pick one row per bucket) rather than a
    plain random sample -- the point is to show the model the scale,
    especially the dominance axis that zero-shot prompting struggles with.
    Picking within a bucket is deterministic via _stable_seed, reusing the
    same hashlib-based seeding convention as _strategy_random.

    Only rows with a usable categorical label and a full non-NaN VAD triple
    are eligible (a demonstration must show the complete required output).
    """
    pool = df[df["session"].isin(train_sessions)]
    pool = pool[pool.apply(is_categorical_usable, axis=1)]
    pool = pool.dropna(subset=["valence", "arousal", "dominance"])
    if len(pool) < n:
        raise ValueError(
            f"Not enough eligible train-session rows ({len(pool)}) for {n} few-shot examples"
        )
    pool = pool.sort_values("dominance").reset_index(drop=True)
    bucket_edges = np.linspace(0, len(pool), n + 1).astype(int)

    examples = []
    for i in range(n):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        bucket = pool.iloc[lo:hi] if hi > lo else pool
        rng = random.Random(_stable_seed(seed, f"few_shot:{i}"))
        row = bucket.iloc[rng.randrange(len(bucket))]
        examples.append(
            {
                "text": row["text"],
                "label": row["emotion"],
                "vad": {
                    "v": round(float(row["valence"]), 1),
                    "a": round(float(row["arousal"]), 1),
                    "d": round(float(row["dominance"]), 1),
                },
            }
        )
    return examples


STRATEGY_REGISTRY: dict[str, Callable[..., list[dict]]] = {
    "none": _strategy_none,
    "window": _strategy_window,
    "random": _strategy_random,
    "sim": _strategy_sim,
    # "llm_select" is intentionally NOT registered here: it's a two-stage
    # (Ollama-calling) selection process handled directly in src/run.py,
    # since a pure (utterance, history, k, **kwargs) -> list[dict] strategy
    # function can't carry model/retry/fallback state back to the caller.
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
