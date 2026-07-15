"""Inference entrypoint: python -m src.run --config configs/<name>.json.

One Ollama chat call per utterance, dual output (categorical label + VAD),
constrained by a JSON schema. Incremental, resumable JSONL output; never
crashes the run on a single bad response (one retry, then record and
continue).
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
)
from src.prompts import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt_c0,
    build_user_prompt_c1,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    return cfg


def apply_smoke(cfg: dict) -> dict:
    cfg["max_eval"] = 20
    print("[smoke] max_eval=20")
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


def load_done_ids(preds_path: Path) -> set[str]:
    """Read existing preds.jsonl, tolerating a corrupt/truncated trailing line.

    Only rows with a successful (non-null) pred_label count as done -- rows
    that failed even after retry (e.g. the Ollama server died mid-run) are
    left out so they get retried on the next invocation, without needing to
    wipe the whole file. src/score.py's load_preds() dedupes by
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
            if uid is not None and rec.get("pred_label") is not None:
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


def process_one(row: dict, history: list[dict], cfg: dict, render_fn, options: dict) -> dict:
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
        cfg["model"], SYSTEM_PROMPT, user_prompt, options, RESPONSE_SCHEMA
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


def write_run_meta(output_dir: Path, cfg: dict) -> None:
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
        "system_prompt": SYSTEM_PROMPT,
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


def dry_run_preview(rows: list[tuple[dict, list[dict]]], render_fn, cfg: dict) -> None:
    print("=" * 80)
    print("SYSTEM PROMPT:")
    print(SYSTEM_PROMPT)
    print("=" * 80)
    print("RESPONSE SCHEMA:")
    print(json.dumps(RESPONSE_SCHEMA, indent=2))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)

    df = load_iemocap(cfg["csv_path"])
    _train, _val, test_session = get_splits(cfg)

    rows = list(iter_eval_rows(df, test_session))
    if cfg.get("max_eval"):
        rows = rows[: cfg["max_eval"]]

    render_fn = resolve_render_fn(cfg)

    if args.dry_run:
        dry_run_preview(rows, render_fn, cfg)
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

    write_run_meta(output_dir, cfg)

    done_ids = load_done_ids(preds_path)
    remaining = [(row, hist) for row, hist in rows if row["utterance_id"] not in done_ids]

    print(
        f"[run] {cfg['experiment_name']}: {len(rows)} total, "
        f"{len(done_ids)} already done, {len(remaining)} remaining"
    )

    options = {"temperature": cfg["temperature"], "seed": cfg["seed"]}
    n_invalid_label = 0
    concurrency = max(1, cfg["concurrency"])

    # Workers only compute (process_one does no file I/O); this thread writes
    # every completed record as it arrives, so preds.jsonl writes stay
    # single-threaded and per-record-flushed regardless of concurrency.
    with open(preds_path, "a") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(process_one, row, history, cfg, render_fn, options)
                for row, history in remaining
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=cfg["experiment_name"],
                unit="utt",
            ):
                record = future.result()
                if record["pred_label"] is None:
                    n_invalid_label += 1
                f.write(json.dumps(record) + "\n")
                f.flush()

    print(
        f"[run] done. {len(remaining)} processed this run, "
        f"{n_invalid_label} with invalid/missing label."
    )


if __name__ == "__main__":
    main()
