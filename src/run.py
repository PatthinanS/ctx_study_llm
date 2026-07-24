"""Inference entrypoint: python -m src.run --config configs/<name>.json.

One Ollama chat call per utterance, constrained by a JSON schema. Output is
dual (categorical label + VAD) by default (cfg["task"] == "both"); --task
vad/cat (or a config's own "task" field) narrows a run to a VAD-only or
categorical-only call with its own system prompt/schema -- see
RESPONSE_SCHEMA_BY_TASK and src/prompts.py's build_system_prompt. Incremental,
resumable JSONL output; never crashes the run on a single bad response (one
retry, then record and continue).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.data import (
    build_context,
    get_splits,
    is_categorical_usable,
    iter_eval_rows,
    load_iemocap,
    select_few_shot_examples,
)
from src.prompts import (
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_CAT,
    RESPONSE_SCHEMA_VAD,
    SELECTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_few_shot_block,
    build_selection_schema,
    build_system_prompt,
    build_user_prompt_c0,
    build_user_prompt_c1,
    build_user_prompt_selection,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

RESPONSE_SCHEMA_BY_TASK = {"both": RESPONSE_SCHEMA, "vad": RESPONSE_SCHEMA_VAD, "cat": RESPONSE_SCHEMA_CAT}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    required = {
        "experiment_name",
        "model",
        "csv_path",
        "splits",
        "condition",
        "context",
        "labels",
        "output_dir",
        "temperature",
        "seed",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Config missing required fields: {sorted(missing)}")
    cfg.setdefault("max_eval", None)
    cfg.setdefault("concurrency", 1)
    cfg.setdefault("eval_split", "test")  # "val" | "test"; C0/C1 configs lack this -> unchanged
    cfg.setdefault("few_shot", None)  # {"n": <int>}; existing configs lack this -> unchanged
    if cfg["few_shot"] is not None:
        if not isinstance(cfg["few_shot"], dict) or "n" not in cfg["few_shot"]:
            raise ValueError('few_shot config must be a dict with an "n" field, e.g. {"n": 4}')
        if not isinstance(cfg["few_shot"]["n"], int) or cfg["few_shot"]["n"] < 1:
            raise ValueError("few_shot.n must be a positive integer")
    cfg.setdefault("task", "both")  # "vad" | "cat" | "both"; existing configs lack this -> unchanged
    if cfg["task"] not in ("vad", "cat", "both"):
        raise ValueError('task must be one of "vad", "cat", "both"')
    return cfg


def apply_smoke(cfg: dict) -> dict:
    cfg["max_eval"] = 20
    print("[smoke] max_eval=20")
    return cfg


def apply_task_override(cfg: dict, task: str | None) -> dict:
    """--task CLI override: if given, replaces cfg["task"] for this invocation."""
    if task is not None:
        cfg["task"] = task
    return cfg


def apply_task_namespacing(cfg: dict) -> dict:
    """Keep vad-only/cat-only runs from colliding with a "both" run of the
    same config (or each other): suffix experiment_name and nest output_dir
    under a task subdirectory, e.g. outputs/vad/c0_llama31_vad/. A no-op for
    task == "both", so the default baseline path is unchanged.
    """
    if cfg["task"] != "both":
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{cfg['task']}"
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / cfg["task"])
    return cfg


def _nan_to_none(x):
    try:
        if x is None:
            return None
        fx = float(x)
        return None if fx != fx else fx  # NaN check without importing math
    except (TypeError, ValueError):
        return None


def resolve_render_fn(cfg: dict) -> Callable[[dict, list[dict]], str]:
    """Build a render_prompt(row, history) -> user_prompt_str closure."""
    condition = cfg["condition"]
    strategy = cfg["context"]["strategy"]
    k = cfg["context"].get("k", 0)
    kwargs = cfg["context"].get("strategy_kwargs", {})

    if condition == "C0":
        def render(row: dict, history: list[dict]) -> str:
            return build_user_prompt_c0(row["text"])

        return render

    if condition == "C1":
        def render(row: dict, history: list[dict]) -> str:
            turns = build_context(strategy, row, history, k, **kwargs)
            return build_user_prompt_c1(turns, row["speaker"], row["text"])

        return render

    raise ValueError(f"Unknown condition '{condition}'")


def _record_succeeded(rec: dict, task: str) -> bool:
    """Whether rec's prediction(s) are non-null for the given task -- the
    signal a "vad"/"cat"/"both" call is actually expected to produce.
    "vad" only ever populates pred_vad (pred_label stays None by design),
    so checking pred_label there would mark every record as failed.
    """
    if task == "vad":
        vad = rec.get("pred_vad")
        return bool(vad) and all(vad.get(dim) is not None for dim in ("v", "a", "d"))
    return rec.get("pred_label") is not None


def load_done_ids(preds_path: Path, task: str = "both") -> set[str]:
    """Read existing preds.jsonl, tolerating a corrupt/truncated trailing line.

    Only rows with a successful (task-appropriate, non-null) prediction count
    as done -- rows that failed even after retry (e.g. the Ollama server died
    mid-run) are left out so they get retried on the next invocation, without
    needing to wipe the whole file. src/score.py's load_preds() dedupes by
    utterance_id (last write wins) to handle the resulting re-attempt rows.
    """
    done: set[str] = set()
    if not preds_path.exists():
        return done
    with open(preds_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("utterance_id")
            if uid is not None and _record_succeeded(rec, task):
                done.add(uid)
    return done


def validate_and_clamp(parsed: dict) -> tuple[str | None, dict]:
    """Return (pred_label_or_None, {'v','a','d'} each float-or-None, clamped to [1,5])."""
    from src.data import LABELS

    label = parsed.get("label") if isinstance(parsed, dict) else None
    pred_label = label if label in LABELS else None

    vad_in = parsed.get("vad") if isinstance(parsed, dict) else None
    vad_out = {}
    for dim in ("v", "a", "d"):
        val = vad_in.get(dim) if isinstance(vad_in, dict) else None
        val = _nan_to_none(val)
        if val is not None:
            val = max(1.0, min(5.0, float(val)))
        vad_out[dim] = val
    return pred_label, vad_out


def call_ollama_once(
    model: str, system: str, user: str, options: dict, schema: dict
) -> tuple[dict | None, str, float]:
    """Single Ollama chat call. Returns (parsed_json_or_None, raw_content, latency_ms).

    Never raises -- callers implement the retry policy.
    """
    import ollama

    start = time.perf_counter()
    raw = ""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,
            options=options,
        )
        raw = response["message"]["content"]
        parsed = json.loads(raw)
    except Exception as e:  # noqa: BLE001 -- must never crash the run
        latency_ms = (time.perf_counter() - start) * 1000
        return None, raw or f"<error: {e}>", latency_ms
    latency_ms = (time.perf_counter() - start) * 1000
    return parsed, raw, latency_ms


def call_ollama_with_retry(
    model: str, system: str, user: str, options: dict, schema: dict
) -> tuple[str | None, dict, float, str]:
    """One retry on any failure or structural invalidity. Never raises.

    Returns (pred_label, pred_vad, latency_ms, raw_response).
    """
    for _attempt in range(2):
        parsed, raw, latency_ms = call_ollama_once(model, system, user, options, schema)
        if parsed is not None:
            pred_label, pred_vad = validate_and_clamp(parsed)
            return pred_label, pred_vad, latency_ms, raw
    # both attempts failed to even parse
    return None, {"v": None, "a": None, "d": None}, latency_ms, raw


def process_one(
    row: dict,
    history: list[dict],
    cfg: dict,
    render_fn,
    options: dict,
    system_prompt: str = SYSTEM_PROMPT,
    response_schema: dict = RESPONSE_SCHEMA,
) -> dict:
    """Run one utterance through Ollama and build its preds.jsonl record.

    Pure compute, no file I/O -- safe to call concurrently from a thread pool.
    """
    user_prompt = render_fn(row, history)
    gold_label = row["emotion"] if is_categorical_usable(row) else None
    gold_vad = {
        "v": _nan_to_none(row["valence"]),
        "a": _nan_to_none(row["arousal"]),
        "d": _nan_to_none(row["dominance"]),
    }

    pred_label, pred_vad, latency_ms, raw = call_ollama_with_retry(
        cfg["model"], system_prompt, user_prompt, options, response_schema
    )

    return {
        "utterance_id": row["utterance_id"],
        "condition": cfg["condition"],
        "model": cfg["model"],
        "gold_label": gold_label,
        "gold_vad": gold_vad,
        "pred_label": pred_label,
        "pred_vad": pred_vad,
        "latency_ms": latency_ms,
        "raw_response": raw,
    }


def _extract_gold_fields(row: dict) -> tuple[str | None, dict]:
    """Gold label/VAD extraction shared by the C2 process functions below.

    Factored out of process_one's inline logic rather than refactoring
    process_one itself, so C0/C1 behavior/code stays untouched.
    """
    gold_label = row["emotion"] if is_categorical_usable(row) else None
    gold_vad = {
        "v": _nan_to_none(row["valence"]),
        "a": _nan_to_none(row["arousal"]),
        "d": _nan_to_none(row["dominance"]),
    }
    return gold_label, gold_vad


def process_one_c2ab(
    row: dict,
    history: list[dict],
    cfg: dict,
    strategy: str,
    k: int,
    kwargs: dict,
    options: dict,
    system_prompt: str = SYSTEM_PROMPT,
    response_schema: dict = RESPONSE_SCHEMA,
) -> dict:
    """C2a ("random") / C2b ("sim") single-call path.

    Pool == history (all prior turns of the current dialogue). Reuses the
    existing C1 template + dual-output call unchanged; adds selected_indices
    (pool indices into history) and pool_size to the record.
    """
    pool_size = len(history)
    turns = build_context(strategy, row, history, k, **kwargs)
    id_to_idx = {h["utterance_id"]: i for i, h in enumerate(history)}
    selected_indices = [id_to_idx[t["utterance_id"]] for t in turns]

    user_prompt = build_user_prompt_c1(turns, row["speaker"], row["text"])
    gold_label, gold_vad = _extract_gold_fields(row)
    pred_label, pred_vad, latency_ms, raw = call_ollama_with_retry(
        cfg["model"], system_prompt, user_prompt, options, response_schema
    )

    return {
        "utterance_id": row["utterance_id"],
        "condition": cfg["condition"],
        "model": cfg["model"],
        "gold_label": gold_label,
        "gold_vad": gold_vad,
        "pred_label": pred_label,
        "pred_vad": pred_vad,
        "latency_ms": latency_ms,
        "raw_response": raw,
        "selected_indices": selected_indices,
        "pool_size": pool_size,
    }


def _validate_selection(parsed, n_sel: int, pool_size: int) -> list[int] | None:
    """Indices must be unique ints in [0, pool_size), exactly n_sel of them."""
    if not isinstance(parsed, dict):
        return None
    sel = parsed.get("selected")
    if not isinstance(sel, list) or len(sel) != n_sel:
        return None
    if any(not isinstance(i, int) or isinstance(i, bool) for i in sel):
        return None
    if len(set(sel)) != len(sel):
        return None
    if any(i < 0 or i >= pool_size for i in sel):
        return None
    return sel


def select_stage1(row: dict, history: list[dict], cfg: dict, k: int, kwargs: dict, options: dict) -> dict:
    """C2c stage 1: one Ollama call to pick n_sel prior-turn indices.

    Returns a dict with selected_indices, fallback, stage1_skipped,
    stage1_latency_ms, stage1_raw_response. Never raises. On repeated
    invalid/failed output, falls back to the last n_sel turns (recency).
    """
    pool_size = len(history)
    n_sel = min(k, pool_size)

    if pool_size <= n_sel:
        return {
            "selected_indices": list(range(pool_size)),
            "fallback": False,
            "stage1_skipped": True,
            "stage1_latency_ms": 0.0,
            "stage1_raw_response": None,
        }

    judge_model = kwargs.get("judge_model", cfg["model"])
    schema = build_selection_schema(n_sel)
    user_prompt = build_user_prompt_selection(history, row["speaker"], row["text"], n_sel)

    latency_ms = 0.0
    raw = ""
    for _attempt in range(2):
        parsed, raw, latency_ms = call_ollama_once(
            judge_model, SELECTION_SYSTEM_PROMPT, user_prompt, options, schema
        )
        indices = _validate_selection(parsed, n_sel, pool_size)
        if indices is not None:
            return {
                "selected_indices": sorted(indices),
                "fallback": False,
                "stage1_skipped": False,
                "stage1_latency_ms": latency_ms,
                "stage1_raw_response": raw,
            }

    fallback_indices = list(range(pool_size - n_sel, pool_size))  # recency
    return {
        "selected_indices": fallback_indices,
        "fallback": True,
        "stage1_skipped": False,
        "stage1_latency_ms": latency_ms,
        "stage1_raw_response": raw,
    }


def process_one_c2c(
    row: dict,
    history: list[dict],
    cfg: dict,
    k: int,
    kwargs: dict,
    options: dict,
    system_prompt: str = SYSTEM_PROMPT,
    response_schema: dict = RESPONSE_SCHEMA,
) -> dict:
    """C2c ("llm_select") two-stage path: LLM-judged selection, then the
    existing dual-output prediction call with the selected turns as context.

    system_prompt/response_schema only affect the stage-2 prediction call --
    stage 1 (select_stage1) always uses SELECTION_SYSTEM_PROMPT, since it
    doesn't output VAD/label and few-shot VAD calibration is irrelevant to it.
    """
    pool_size = len(history)
    stage1 = select_stage1(row, history, cfg, k, kwargs, options)
    selected_indices = sorted(stage1["selected_indices"])
    turns = [history[i] for i in selected_indices]

    user_prompt = build_user_prompt_c1(turns, row["speaker"], row["text"])
    gold_label, gold_vad = _extract_gold_fields(row)
    pred_label, pred_vad, stage2_latency_ms, raw = call_ollama_with_retry(
        cfg["model"], system_prompt, user_prompt, options, response_schema
    )

    return {
        "utterance_id": row["utterance_id"],
        "condition": cfg["condition"],
        "model": cfg["model"],
        "gold_label": gold_label,
        "gold_vad": gold_vad,
        "pred_label": pred_label,
        "pred_vad": pred_vad,
        "latency_ms": stage2_latency_ms,
        "raw_response": raw,
        "selected_indices": selected_indices,
        "pool_size": pool_size,
        "fallback": stage1["fallback"],
        "stage1_skipped": stage1["stage1_skipped"],
        "stage1_latency_ms": stage1["stage1_latency_ms"],
        "stage2_latency_ms": stage2_latency_ms,
        "stage1_raw_response": stage1["stage1_raw_response"],
    }


def resolve_c2_process_fn(
    cfg: dict,
    options: dict,
    system_prompt: str = SYSTEM_PROMPT,
    response_schema: dict = RESPONSE_SCHEMA,
):
    """Build a (row, history) -> record closure for condition=="C2",
    dispatching on context.strategy."""
    strategy = cfg["context"]["strategy"]
    k = cfg["context"].get("k", 4)
    kwargs = dict(cfg["context"].get("strategy_kwargs", {}))

    if strategy == "random":
        kwargs.setdefault("seed", cfg["seed"])
        return lambda row, history: process_one_c2ab(
            row, history, cfg, strategy, k, kwargs, options, system_prompt, response_schema
        )
    if strategy == "sim":
        return lambda row, history: process_one_c2ab(
            row, history, cfg, strategy, k, kwargs, options, system_prompt, response_schema
        )
    if strategy == "llm_select":
        return lambda row, history: process_one_c2c(
            row, history, cfg, k, kwargs, options, system_prompt, response_schema
        )
    raise ValueError(f"Unknown C2 context.strategy '{strategy}'")


def write_run_meta(output_dir: Path, cfg: dict, system_prompt: str = SYSTEM_PROMPT) -> None:
    try:
        version_out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        )
        ollama_version = version_out.stdout.strip() or version_out.stderr.strip()
    except Exception:
        ollama_version = None

    try:
        git_out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=10,
        )
        git_commit = git_out.stdout.strip() if git_out.returncode == 0 else None
    except Exception:
        git_commit = None

    meta = {
        "experiment_name": cfg["experiment_name"],
        "config": cfg,
        "system_prompt": system_prompt,
        "model": cfg["model"],
        "ollama_version": ollama_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
    }
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def ensure_ollama_reachable(model: str) -> None:
    """Fail fast with an actionable error if the Ollama server isn't up.

    Never called on the --dry-run path (that path must stay Ollama-free).
    Setup/process management is setup.sh's job, not run.py's -- this is
    purely a clear-error check, not a self-healing retry/launch.
    """
    import ollama

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        response = ollama.list()
    except Exception as e:  # noqa: BLE001
        print(
            f"ERROR: Ollama server not reachable at {host}.\n"
            f"  ({e})\n"
            "  Run ./setup.sh, or start it manually with `ollama serve`.",
            file=sys.stderr,
        )
        sys.exit(1)

    available = {m.get("model") or m.get("name") for m in response.get("models", [])}
    if model not in available:
        print(
            f"WARNING: model '{model}' not found locally. "
            f"Run `ollama pull {model}` or ./setup.sh before this finishes.",
            file=sys.stderr,
        )


def dry_run_preview(
    rows: list[tuple[dict, list[dict]]],
    render_fn,
    cfg: dict,
    system_prompt: str = SYSTEM_PROMPT,
    response_schema: dict = RESPONSE_SCHEMA,
) -> None:
    print("=" * 80)
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("=" * 80)
    print("RESPONSE SCHEMA:")
    print(json.dumps(response_schema, indent=2))
    print("=" * 80)

    samples = []
    # Guarantee one first-utterance (empty-history) case for C1 degradation.
    first_utt = next((r for r in rows if not r[1]), None)
    if first_utt is not None:
        samples.append(("first-utterance (empty context)", first_utt))
    mid_dialogue = next((r for r in rows if len(r[1]) >= 1), None)
    if mid_dialogue is not None and mid_dialogue is not first_utt:
        samples.append(("mid-dialogue (with context)", mid_dialogue))
    for row, history in rows:
        if len(samples) >= 3:
            break
        if (row, history) not in [s[1] for s in samples]:
            samples.append(("additional case", (row, history)))

    for label, (row, history) in samples[:3]:
        print(f"--- {label} | utterance_id={row['utterance_id']} ---")
        print(render_fn(row, history))
        print()


def _pick_dry_run_samples_c2(rows: list[tuple[dict, list[dict]]], k: int):
    samples = []
    small_pool = next((r for r in rows if len(r[1]) <= k), None)
    if small_pool is not None:
        samples.append(("pool_size <= n_sel", small_pool))
    big_pool = next((r for r in rows if len(r[1]) >= 20 and r is not small_pool), None)
    if big_pool is not None:
        samples.append(("20+ prior turns", big_pool))
    for row, history in rows:
        if len(samples) >= 3:
            break
        if (row, history) not in [s[1] for s in samples]:
            samples.append(("additional case", (row, history)))
    return samples[:3]


def dry_run_preview_c2(
    rows: list[tuple[dict, list[dict]]],
    cfg: dict,
    system_prompt: str = SYSTEM_PROMPT,
    response_schema: dict = RESPONSE_SCHEMA,
) -> None:
    strategy = cfg["context"]["strategy"]
    k = cfg["context"].get("k", 4)
    kwargs = dict(cfg["context"].get("strategy_kwargs", {}))

    print("=" * 80)
    print("SYSTEM PROMPT (stage 2 / prediction):")
    print(system_prompt)
    print("=" * 80)
    print("RESPONSE SCHEMA (stage 2 / prediction):")
    print(json.dumps(response_schema, indent=2))
    print("=" * 80)
    if strategy == "llm_select":
        print("SELECTION SYSTEM PROMPT (stage 1):")
        print(SELECTION_SYSTEM_PROMPT)
        print("=" * 80)

    for label, (row, history) in _pick_dry_run_samples_c2(rows, k):
        pool_size = len(history)
        n_sel = min(k, pool_size)
        print(f"--- {label} | utterance_id={row['utterance_id']} | pool_size={pool_size} | n_sel={n_sel} ---")

        if strategy == "llm_select":
            if pool_size <= n_sel:
                print("[stage 1 SKIPPED: pool_size <= n_sel, all turns auto-selected]")
                turns = history
            else:
                print("STAGE 1 PROMPT:")
                print(build_user_prompt_selection(history, row["speaker"], row["text"], n_sel))
                print("STAGE 1 SCHEMA:")
                print(json.dumps(build_selection_schema(n_sel), indent=2))
                print(
                    "[NOTE: --dry-run makes no Ollama calls; the stage-2 context below uses "
                    "a recency placeholder selection for illustration, not real stage-1 output]"
                )
                turns = history[-n_sel:]
            print("STAGE 2 PROMPT:")
        elif strategy == "random":
            kw = dict(kwargs)
            kw.setdefault("seed", cfg["seed"])
            turns = build_context("random", row, history, k, **kw)
            print("RENDERED PROMPT:")
        elif strategy == "sim":
            turns = build_context("sim", row, history, k, **kwargs)
            print("RENDERED PROMPT:")
        else:
            raise ValueError(f"Unknown C2 context.strategy '{strategy}'")

        print(build_user_prompt_c1(turns, row["speaker"], row["text"]))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--task",
        choices=["vad", "cat", "both"],
        default=None,
        help="Override the config's task: \"vad\"/\"cat\" split the single dual-output call into "
        "a VAD-only or categorical-only call with its own system prompt/schema; default is the "
        "config's own task field, or \"both\" (today's behavior) if it has none.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_task_override(cfg, args.task)
    cfg = apply_task_namespacing(cfg)
    if args.smoke:
        cfg = apply_smoke(cfg)

    df = load_iemocap(cfg["csv_path"])
    train_sessions, val_session, test_session = get_splits(cfg)
    session = val_session if cfg["eval_split"] == "val" else test_session

    rows = list(iter_eval_rows(df, session))
    if cfg.get("max_eval"):
        rows = rows[: cfg["max_eval"]]

    condition = cfg["condition"]

    few_shot_cfg = cfg.get("few_shot")
    few_shot_block = None
    if few_shot_cfg:
        examples = select_few_shot_examples(df, train_sessions, few_shot_cfg["n"], cfg["seed"])
        few_shot_block = build_few_shot_block(examples, task=cfg["task"])
    system_prompt = build_system_prompt(few_shot_block, cfg["task"])
    response_schema = RESPONSE_SCHEMA_BY_TASK[cfg["task"]]

    if args.dry_run:
        if condition in ("C0", "C1"):
            render_fn = resolve_render_fn(cfg)
            dry_run_preview(rows, render_fn, cfg, system_prompt, response_schema)
        elif condition == "C2":
            dry_run_preview_c2(rows, cfg, system_prompt, response_schema)
        else:
            raise ValueError(f"Unknown condition '{condition}'")
        return

    ensure_ollama_reachable(cfg["model"])

    output_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.jsonl"

    if preds_path.exists():
        try:
            answer = input(
                f"{preds_path} already exists. Remove it and start fresh? [y/N] "
            ).strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            preds_path.unlink()
            print(f"[run] removed {preds_path}, starting fresh")

    write_run_meta(output_dir, cfg, system_prompt)

    done_ids = load_done_ids(preds_path, cfg["task"])
    remaining = [(row, hist) for row, hist in rows if row["utterance_id"] not in done_ids]

    print(
        f"[run] {cfg['experiment_name']}: {len(rows)} total, "
        f"{len(done_ids)} already done, {len(remaining)} remaining"
    )

    options = {"temperature": cfg["temperature"], "seed": cfg["seed"]}
    n_invalid = 0
    concurrency = max(1, cfg["concurrency"])

    if condition in ("C0", "C1"):
        render_fn = resolve_render_fn(cfg)
        submit_fn = lambda row, history: process_one(
            row, history, cfg, render_fn, options, system_prompt, response_schema
        )
    elif condition == "C2":
        submit_fn = resolve_c2_process_fn(cfg, options, system_prompt, response_schema)
    else:
        raise ValueError(f"Unknown condition '{condition}'")

    # Workers only compute (process_one/submit_fn do no file I/O); this thread
    # writes every completed record as it arrives, so preds.jsonl writes stay
    # single-threaded and per-record-flushed regardless of concurrency.
    with open(preds_path, "a") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(submit_fn, row, history)
                for row, history in remaining
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=cfg["experiment_name"],
                unit="utt",
            ):
                record = future.result()
                if not _record_succeeded(record, cfg["task"]):
                    n_invalid += 1
                f.write(json.dumps(record) + "\n")
                f.flush()

    print(
        f"[run] done. {len(remaining)} processed this run, "
        f"{n_invalid} with invalid/missing prediction."
    )


if __name__ == "__main__":
    main()
