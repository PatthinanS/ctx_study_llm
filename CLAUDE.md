# CLAUDE.md

Working notes for this repo — read before making non-trivial changes. Not a rehash of README.md; that's the user-facing usage doc.

## What this is

The LLM-prompting leg of an ERC (emotion recognition in conversation) research study: inference-only over IEMOCAP via Ollama, no training. Sibling to the PLM (fine-tuning) repo `ContextStudy_NewResearch` — results must stay comparable, so splits and dialogue reconstruction match that repo exactly. Runs on a **remote Linux box with no root/sudo access** via `./setup.sh` (self-contained: one conda env provides both the Ollama server binary and the Python deps; starts the server; pulls the default model). Originally scoped for local macOS too (Ollama via Homebrew works fine there).

Python environment is a **conda env** (`environment.yml`, default name `venv`) rather than Python's built-in `venv` tool — switched to conda because the remote box has conda available but no sudo, and conda envs are fully user-space (no root needed for either conda itself, typically installed via Miniconda into `$HOME`, or for env creation).

## No-root Ollama install: use conda-forge, not Ollama's own installer

Two problems ruled out Ollama's own `install.sh`/prebuilt binary on the remote box, in order of discovery:

1. **Needs sudo.** It requires root for installing into `/usr/local`, chown-ing to root, and creating a systemd service running as a dedicated `ollama` user — none of that is actually required just to run `ollama serve` as the current user. (Verified against the real `install.sh` source, fetched directly, before ruling this out.)
2. **Even side-stepping sudo by extracting the release tarball into a user-writable prefix** (what an earlier version of `setup.sh` did — download `ollama-linux-${arch}.tar.zst`/`.tgz` from `ollama.com/download` and `tar`-extract into `~/.local/ollama`) **still failed on the remote box** with `libc.so.6: version 'GLIBC_2.28' not found` — Ollama's official binary requires a newer glibc than the remote has, and you can't upgrade system glibc without root either.

The fix: install via **`conda-forge::ollama`** instead (confirmed to exist — `curl -s "https://api.anaconda.org/search?name=ollama"` lists it under the `conda-forge` channel with many versions). conda-forge builds target a much older baseline glibc for portability, so it runs on the remote's older libc. This is now just a line in `environment.yml`'s conda `dependencies:` — `setup.sh` no longer does any manual curl/tar/zstd archive handling at all; `conda env create -f environment.yml` installs the `ollama` CLI/server binary and all Python deps (including the separate `ollama` **Python client** from pip — different package, same name, no conflict) in one step.

If this ever needs revisiting: don't reintroduce the tarball-extraction approach as the primary path — it's a strictly worse fit for this remote environment than conda-forge on both the sudo axis and the glibc axis.

## `ollama` version is pinned in `environment.yml` — don't casually bump it

`environment.yml` pins `ollama=0.22.0=cpu_hacd46da_0` rather than leaving the dependency unconstrained. This is deliberate, found the hard way on the remote box (2026-07-14):

conda-forge's `ollama` **0.30.x** builds — confirmed for both the `cpu_hacd46da_0` and `cuda_129`/`cuda_130` build variants — are missing the bundled `llama-server` runner binary (the actual llama.cpp inference engine; the main `ollama` binary is just the Go orchestrator that spawns it per-model on first use). This is a known upstream packaging regression, not specific to this repo or this remote box — see [ollama/ollama#16535](https://github.com/ollama/ollama/issues/16535), [#16643](https://github.com/ollama/ollama/issues/16643), and the parallel Homebrew report [Homebrew/homebrew-core#285982](https://github.com/Homebrew/homebrew-core/issues/285982). Symptom: `ollama serve` starts fine and `ollama list`/`ollama pull` work fine (they don't need the runner), but any actual inference call (`ollama run`, or `ollama.chat()` from `src/run.py`) fails with:

```
Error: 500 Internal Server Error: error starting llama-server: llama-server binary not found (checked: ...). Run 'cmake -S llama/server --preset cpu && cmake --build --preset cpu' first
```

**`ollama serve` starting successfully is not evidence the install is healthy** — the runner is only spawned lazily on the first real inference request for a given model, so this failure mode can hide until you're well into a run. Don't use `find <prefix> -iname "*llama-server*"` as a health check either — it produced false negatives even against versions that turned out to work fine (the runner may only materialize on first use, not at install time); the only reliable test is an actual `ollama run <model> "..."` call.

The fix was downgrading to `0.22.0` (confirmed working via a real `ollama run` call, glibc floor `>=2.17` so compatible with older remote boxes too). If bumping this version in the future, verify with a real inference call on the target box first, not just `ollama --version` or `find`.

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
./setup.sh                                              # bootstrap everything (idempotent, no sudo)
conda activate venv
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
