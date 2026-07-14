# ContextStudy_LLM — LLM Prompting Leg

This is the LLM-prompting leg of an ERC (emotion recognition in conversation) research study, run via [Ollama](https://ollama.com). It is inference-only — no training — and is a sibling to the PLM (fine-tuning) leg at `ContextStudy_NewResearch`. Two conditions are implemented: **C0** (target utterance only, no context) and **C1** (last k=3 prior turns of the same dialogue as context). C2 (retrieval-based context) and C3 (LoRA factorial) are designed to slot in later without restructuring — see "Extending" below. Designed to run unattended on a remote Linux box (`./setup.sh` provisions everything, including the Ollama server itself); local macOS development also works.

Data conventions match the `ContextStudy_NewResearch` PLM repo exactly where it matters for comparability: leave-one-session-out splits (test=Session5, val=Session4, train=Sessions1-3) and the same source CSV/dialogue reconstruction (group by `(session, dialog)`, sort by `start_time`). The 6-way categorical filter (`LABELS = [ang, hap, exc, neu, sad, fru]`, excluding rows with `emotion` in `{xxx, sur, fea, dis, oth, ""}` or missing) is new to this repo — the PLM repo is VAD-regression only and has no categorical component. Dimensional (VAD) metrics are computed over the full test split (every row with a non-null gold VAD triple, since Session5 has a small number of rows with missing VAD annotations), while categorical metrics are computed only over the subset of rows with a usable gold label AND a valid model prediction.

## Setup

Requires `conda` (e.g. Miniconda) already installed — Miniconda installs entirely into `$HOME`, no root needed: https://docs.conda.io/en/latest/miniconda.html

One-command bootstrap (idempotent, safe to re-run), **no root/sudo required anywhere**:

```bash
./setup.sh
```

This creates/updates a conda env named `venv` from `environment.yml` — which installs **both** the Ollama server binary (the `conda-forge::ollama` package) **and** the Python dependencies in one shot — then starts the Ollama server in the background and pulls the default model.

Ollama comes from conda-forge rather than Ollama's own installer/binary deliberately: their official binary requires a fairly recent glibc (e.g. `GLIBC_2.28+`), which older remote boxes may not have and which you can't upgrade without root. conda-forge's build targets a much older baseline glibc for portability, so it runs on older systems too.

The `ollama` version in `environment.yml` is pinned deliberately (`0.22.0`, not latest) — conda-forge's `0.30.x` builds are missing the bundled `llama-server` runner binary (an upstream packaging bug), so inference calls fail with "llama-server binary not found" even though `ollama serve` itself starts fine. See `CLAUDE.md` before bumping this version.

Override with env vars if needed: `OLLAMA_MODEL=<tag>`, `CONDA_ENV_NAME=<name>` (default `venv`).

To do it manually instead:

```bash
conda env create -f environment.yml   # or: conda env update -f environment.yml --prune
conda activate venv

ollama serve &          # start the local server if not already running
ollama pull mistral:7b-instruct-q4_K_M
```

`src/run.py` fails fast with a clear error if the Ollama server isn't reachable when a real run starts (`--dry-run` never needs it).

## Data placement

Copy the merged IEMOCAP CSV into `data/iemocap/iemocap_merged_all.csv` (the `data/` directory is gitignored):

```bash
mkdir -p data/iemocap
cp /path/to/iemocap_merged_all.csv data/iemocap/
```

Required columns: `session, dialog, utterance_id, speaker, start_time, end_time, text, emotion, valence, arousal, dominance`.

## Usage

With the `venv` conda env active (`conda activate venv`):

```bash
# Dry run: print 3 fully-rendered prompts (incl. a first-utterance C1 case
# with empty context) and exit without calling Ollama.
python -m src.run --config configs/c0_mistral.json --dry-run
python -m src.run --config configs/c1_mistral.json --dry-run

# Smoke test: 20 utterances end-to-end through real Ollama.
python -m src.run --config configs/c0_mistral.json --smoke
python -m src.run --config configs/c1_mistral.json --smoke

# Full runs.
python -m src.run --config configs/c0_mistral.json
python -m src.run --config configs/c1_mistral.json

# Score.
python -m src.score --run outputs/c0_mistral
python -m src.score --run outputs/c1_mistral
```

Runs are resumable: interrupting and rerunning the same command skips utterance_ids already present in `outputs/<experiment_name>/preds.jsonl`.

## Output layout

```
outputs/<experiment_name>/
  preds.jsonl    # one JSON record per utterance
  run_meta.json  # full config, system prompt, model, ollama version, timestamp, git commit
  metrics.json   # written by src.score, categorical + dimensional blocks
```

## Testing

```bash
pytest tests/
```

No Ollama connection is required for the test suite. The split-parity test (`Session5` test split == 2195 rows) is skipped automatically if `data/iemocap/iemocap_merged_all.csv` is not present.

## Extending

- **C2 (retrieval)**: add a new function to `STRATEGY_REGISTRY` in `src/data.py` with the same signature as `_strategy_window`, then add a config with `condition: "C1"` and `context.strategy: "retrieval"` — the existing C1 prompt template is reused unchanged.
- **C3 (LoRA factorial)**: add a new config with a different `model` tag (an Ollama model built from a LoRA-adapted GGUF). No code changes are needed — `model` is threaded straight into the Ollama call, and `run_meta.json` already logs the exact model string per run for traceability.
