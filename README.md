# ContextStudy_LLM — LLM Prompting Leg

This is the LLM-prompting leg of an ERC (emotion recognition in conversation) research study, run via [Ollama](https://ollama.com). It is inference-only — no training — and is a sibling to the PLM (fine-tuning) leg at `ContextStudy_NewResearch`. Three conditions are implemented: **C0** (target utterance only, no context), **C1** (last k=4 prior turns of the same dialogue as context), and **C2** (retrieval-augmented context, three variants: `random`, `sim` cosine-similarity, `llm_select` LLM-judged selection — see "Extending" below). C3 (LoRA factorial) is designed to slot in later without restructuring. Designed to run unattended on a remote Linux box (`./setup.sh` provisions everything, including the Ollama server itself); local macOS development also works.

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

### C2 (retrieval-augmented context)

```bash
# Dry run (c2_llm also previews the stage-1 selection prompt/schema).
python -m src.run --config configs/c2_random_llama31_val.json --dry-run
python -m src.run --config configs/c2_sim_llama31_val.json --dry-run
python -m src.run --config configs/c2_llm_llama31_val.json --dry-run

# Smoke / full runs, same pattern as C0/C1.
python -m src.run --config configs/c2_random_llama31_val.json --smoke
python -m src.run --config configs/c2_sim_llama31_val.json --smoke
python -m src.run --config configs/c2_llm_llama31_val.json --smoke

python -m src.run --config configs/c2_random_llama31_val.json
python -m src.run --config configs/c2_sim_llama31_val.json
python -m src.run --config configs/c2_llm_llama31_val.json

# Score, plus selection-overlap analysis (Jaccard vs. recency and vs. other
# C2 runs, matched by utterance_id) for any run carrying selected_indices.
python -m src.score --run outputs/c2_random_llama31_val
python -m src.score --run outputs/c2_sim_llama31_val
python -m src.score --run outputs/c2_llm_llama31_val
python -m src.score --run outputs/c2_llm_llama31_val --compare-selections outputs/c2_sim_llama31_val
python -m src.score --run outputs/c2_sim_llama31_val --compare-selections
```

`c2_sim` fetches `xlm-roberta-base` weights from the Hugging Face Hub on first use (needs outbound network access; not cached by `./setup.sh`).

The `_val` configs run against the val session (Session4) via a top-level `"eval_split": "val"` config field (default `"test"` if omitted, matching C0/C1's existing behavior); `_test` twins of all three configs run against Session5.

### Few-shot (demo)

An optional top-level `"few_shot": {"n": 4}` config field appends a fixed block of `n` text→(label, VAD) demonstrations to the system prompt, in the exact `RESPONSE_SCHEMA` output shape (`{"label": ..., "vad": {"v": ..., "a": ..., "d": ...}}`), so the model sees the required format and the 1.0-5.0 scale together before annotating. Examples are drawn only from `splits.train_sessions` (never val/test), spread across the dominance range (one per dominance-sorted bucket, picked deterministically from `seed`) rather than sampled at random. Configs without a `few_shot` field are unaffected — this is purely additive.

```bash
# Dry run: the EXAMPLES block appears in the printed system prompt for the
# fewshot configs and is absent for the zero-shot ones.
python -m src.run --config configs/c0_fewshot_llama31_val_demo.json --dry-run
python -m src.run --config configs/c1_fewshot_llama31_val_demo.json --dry-run
python -m src.run --config configs/c2_llm_fewshot_llama31_val_demo.json --dry-run

# Smoke (20 utterances), zero-shot vs. few-shot pairs, all on val (Session4).
python -m src.run --config configs/c0_llama31_val_demo.json --smoke
python -m src.run --config configs/c0_fewshot_llama31_val_demo.json --smoke
python -m src.run --config configs/c1_llama31_val_demo.json --smoke
python -m src.run --config configs/c1_fewshot_llama31_val_demo.json --smoke
python -m src.run --config configs/c2_llm_llama31_val_demo.json --smoke
python -m src.run --config configs/c2_llm_fewshot_llama31_val_demo.json --smoke

python -m src.score --run outputs/c0_llama31_val_demo
python -m src.score --run outputs/c0_fewshot_llama31_val_demo
python -m src.score --run outputs/c1_llama31_val_demo
python -m src.score --run outputs/c1_fewshot_llama31_val_demo
python -m src.score --run outputs/c2_llm_llama31_val_demo
python -m src.score --run outputs/c2_llm_fewshot_llama31_val_demo
```

Each zero-shot/few-shot pair shares every config field except `few_shot`, so `--smoke`'s deterministic first-20-rows slice means both runs in a pair score the identical 20 utterances — a clean paired comparison. 20 samples is far too few for a reliable Pearson r; treat this as a "did anything move at all" smoke test, not a result.

### Task split: VAD-only vs. categorical-only calls

By default every run makes a single dual-output Ollama call, asking for a categorical emotion label *and* a Valence/Arousal/Dominance (VAD) triple together. A `--task {vad,cat,both}` flag (or an equivalent top-level `"task"` config field) narrows a run to one output only, with its own system prompt and JSON schema — useful for testing whether asking for both jointly changes what the model predicts, versus asking for each separately. `both` (the default) is byte-for-byte the original prompt/schema; no existing config or command needs to change.

VAD/arousal/dominance definitions used in the VAD-only (and dual) prompt:
- **Valence (v)**: how positive or negative the emotional tone is — the pleasantness or unpleasantness of the speaker's affective state. 1.0 = very negative, 5.0 = very positive.
- **Arousal (a)**: the intensity of activation in the speaker's affective state — how calm/passive versus excited/energized they sound. 1.0 = very calm, 5.0 = very activated.
- **Dominance (d)**: the degree of control or power conveyed by the speaker's affective state — how submissive/controlled versus in-control/dominant they sound. 1.0 = very submissive, 5.0 = very dominant.

`--task` never requires a new config file: it overrides whatever the config says (or the `"both"` default) for that one invocation. Whenever the effective task isn't `"both"`, output is automatically namespaced under `outputs/{vad,cat}/<experiment_name>_{vad,cat}/` so it can't collide with a `"both"` run of the same config, or with the other task's run.

```bash
# Dry run: system prompt/schema shown match the requested task only.
python -m src.run --config configs/c0_llama31_val_demo.json --task vad --dry-run
python -m src.run --config configs/c0_llama31_val_demo.json --task cat --dry-run

# Smoke (20 utterances).
python -m src.run --config configs/c0_llama31_val_demo.json --task vad --smoke
python -m src.run --config configs/c0_llama31_val_demo.json --task cat --smoke

python -m src.score --run outputs/vad/c0_llama31_val_demo_vad
python -m src.score --run outputs/cat/c0_llama31_val_demo_cat
```

A `vad`-task run's `metrics.json` will have a populated `dimensional` block and an empty/nan `categorical` block (no `label` was ever requested); a `cat`-task run is the mirror image.

## Output layout

```
outputs/<experiment_name>/
  preds.jsonl    # one JSON record per utterance
  run_meta.json  # full config, system prompt, model, ollama version, timestamp, git commit

eval/<experiment_name>/
  metrics.json   # written by src.score, categorical + dimensional blocks
```

`src.score` reads from `outputs/<experiment_name>/` but writes derived scoring results separately, to `eval/<experiment_name>/` — override the eval location with `--eval-dir <path>` if needed.

## Testing

```bash
pytest tests/
```

No Ollama connection is required for the test suite. The split-parity test (`Session5` test split == 2195 rows) is skipped automatically if `data/iemocap/iemocap_merged_all.csv` is not present.

## Extending

- **C2 (retrieval, implemented)**: `condition: "C2"` with `context.strategy` one of `"random"` (C2a, deterministic uniform sample via a hashlib-seeded RNG), `"sim"` (C2b, cosine similarity over `xlm-roberta-base` embeddings — encoder name configurable via `context.strategy_kwargs.encoder`), or `"llm_select"` (C2c, a two-stage flow: one Ollama call selects which prior turns to keep, judged by emotional relevance not topical similarity, then the existing C1 template + dual-output call runs on the selected context; judge model configurable via `context.strategy_kwargs.judge_model`, default same as the prediction model). Every C2 record additionally carries `selected_indices`/`pool_size`; C2c records also carry `fallback`/`stage1_skipped`/`stage1_latency_ms`/`stage2_latency_ms`/`stage1_raw_response`. `src.score --compare-selections` computes Jaccard overlap between any two C2 runs' selections (and against a recency baseline).
- **C3 (LoRA factorial)**: add a new config with a different `model` tag (an Ollama model built from a LoRA-adapted GGUF). No code changes are needed — `model` is threaded straight into the Ollama call, and `run_meta.json` already logs the exact model string per run for traceability.
