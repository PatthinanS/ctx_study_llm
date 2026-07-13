# CLAUDE.md

Working notes for this repo — read before making non-trivial changes. Not a rehash of README.md; that's the user-facing usage doc.

## What this is

The LLM-prompting leg of an ERC (emotion recognition in conversation) research study: inference-only over IEMOCAP via Ollama, no training. Sibling to the PLM (fine-tuning) repo `ContextStudy_NewResearch` — results must stay comparable, so splits and dialogue reconstruction match that repo exactly. Runs on a remote Linux box via `./setup.sh` (self-contained: installs the Ollama server binary, not just the Python client, starts it, pulls the default model). Originally scoped for local macOS too; `setup.sh` branches on `uname` for both.

## Data conventions that must not drift

- `session` values stay plain strings (`"Session1"`.."Session5") everywhere — never coerce to int. The sibling PLM repo has a latent dtype bug around exactly this (an `if dtype == object` check that silently no-ops under pandas' newer string dtype); this repo avoids the whole class of bug by never converting.
- `src/data.py`'s `load_iemocap` must never `dropna` on valence/arousal/dominance at load time. The PLM repo does, and it's tempting to mirror that — don't. It would drop `Session5` from 2195 rows to 2170, breaking the split-parity test (`tests/test_data.py::test_session5_split_row_count`) and silently removing valid context turns (rows with missing VAD can still be perfectly good context/text). NaN-gold-VAD rows are excluded only at dimensional-scoring time, in `src/score.py::build_dimensional_arrays` — categorical scoring and context construction are unaffected by them.
- `LABELS` and `EXCLUDED_EMOTIONS` in `src/data.py` are the single source of truth for the 6-way categorical filter. This categorical logic is new to this repo — the PLM repo is pure VAD regression and has no equivalent to port from.

## Architecture / extensibility seam

`condition` (which prompt template — `src/prompts.py`) is deliberately decoupled from `context.strategy` (which turns get selected — `src/data.py`'s `STRATEGY_REGISTRY`). This is why config has two separate fields instead of one combined mode string:

- **C2 (retrieval)**, when added: one new `STRATEGY_REGISTRY` entry in `data.py` with the same `(utterance, history, k, **kwargs) -> list[dict]` signature as `_strategy_window`. Reuses the existing C1 prompt template unchanged — no `prompts.py` edit needed.
- **C3 (LoRA factorial)**, when added: just a new `model` string in a config pointing at an Ollama model built from a LoRA adapter. No code change anywhere — `model` already threads straight into `ollama.chat(model=cfg["model"], ...)`, and `run_meta.json` already logs it per run.

Don't collapse `condition` and `context.strategy` back into one field — that's what keeps both extensions restructuring-free.

## Invariants to preserve

- `src/run.py`'s inference loop must never raise on a bad/missing/malformed Ollama response. One retry (`call_ollama_with_retry`), then record whatever's salvageable (nulls where invalid) and continue. A single bad utterance must never kill a multi-hour remote run.
- `--dry-run` must stay completely Ollama-free (no import, no network call) — it's the fast local sanity check for prompt rendering. `ensure_ollama_reachable` is called in `main()` only after the dry-run early return.
- `preds.jsonl` writes are flushed per record (resume safety — a killed process mid-run must not lose more than the in-flight utterance).

## Commands

```
./setup.sh                                              # bootstrap everything (idempotent)
python -m src.run --config configs/c0_mistral.json --dry-run
python -m src.run --config configs/c1_mistral.json --dry-run
python -m src.run --config configs/c0_mistral.json --smoke
python -m src.run --config configs/c1_mistral.json --smoke
python -m src.run --config configs/c0_mistral.json
python -m src.run --config configs/c1_mistral.json
python -m src.score --run outputs/c0_mistral
python -m src.score --run outputs/c1_mistral
```

## Testing

`pytest tests/` needs no live Ollama server — `ollama.chat`/`ollama.list` are mocked where exercised (see `tests/test_run_resume.py` for the `unittest.mock.patch("ollama.chat", ...)` pattern to follow for any new Ollama-touching code). The split-parity test (`Session5` == 2195 rows) auto-skips if `data/iemocap/iemocap_merged_all.csv` isn't present.
