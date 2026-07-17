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

`condition` (which prompt template/record shape — `src/prompts.py`) is deliberately decoupled from `context.strategy` (which turns get selected — `src/data.py`'s `STRATEGY_REGISTRY`). This is why config has two separate fields instead of one combined mode string.

- **C2 (retrieval), implemented**: `condition: "C2"` is a genuinely new, distinct condition value — **not** `condition: "C1"` with a new strategy, despite what an earlier version of this file and README.md said. Two things forced that: every C2 record needs extra `preds.jsonl` fields (`selected_indices`, `pool_size`) that C0/C1 records don't carry, and C2c's two-Ollama-call-per-utterance flow doesn't fit the single-string `render_fn(row, history) -> str` contract that `resolve_render_fn`/`process_one` are built around. `src/run.py`'s `main()` branches on `condition` in two places (the `--dry-run` block and the thread-pool submit block) to pick between the untouched C0/C1 path and the new C2 path (`resolve_c2_process_fn` → `process_one_c2ab` for `random`/`sim`, `process_one_c2c` for `llm_select`) — C0/C1 code is not modified.
  - `random`/`sim` are ordinary `STRATEGY_REGISTRY` entries (`_strategy_random`, `_strategy_sim` in `data.py`), reusing the existing C1 template unchanged (`build_user_prompt_c1`). `selected_indices` is derived generically from any registered strategy's return value by mapping each returned turn's `utterance_id` back to its position in `history` — no need for a strategy function to return indices itself.
  - `llm_select` (C2c) is deliberately **not** a `STRATEGY_REGISTRY` entry — its stage-1 selection call needs `cfg["model"]`/`judge_model`/retry/fallback state that a pure `(utterance, history, k, **kwargs) -> list[dict]` function can't carry back to the caller, so it's implemented directly in `run.py` (`select_stage1`, `process_one_c2c`).
  - `_strategy_sim`'s embedding cache (`_EMBED_CACHE` in `data.py`) is lock-guarded (`_EMBED_LOCK`) because `main()`'s `ThreadPoolExecutor` can call the strategy for multiple utterances concurrently. "Embed each dialogue once" only means "never re-embed an already-embedded turn" — a strategy call only ever sees prior turns + the current utterance, never the full future dialogue, so there's no way to batch a whole dialogue upfront. `_encode_batch` is the sole function touching `torch`/`transformers` (imported lazily inside it), and is the seam tests monkeypatch to avoid any real model load.
  - `c2_sim` fetches `xlm-roberta-base` from the Hugging Face Hub over HTTPS on first use — unlike Ollama models, this isn't provisioned by `setup.sh`. If the remote box lacks outbound network access, this fails on first `c2_sim` invocation; verify with a real `AutoModel.from_pretrained("xlm-roberta-base")` call before relying on it there (same "verify with a real call, not a presence check" lesson as the Ollama runner-binary issue above).
- **C3 (LoRA factorial)**, when added: just a new `model` string in a config pointing at an Ollama model built from a LoRA adapter. No code change anywhere — `model` already threads straight into `ollama.chat(model=cfg["model"], ...)`, and `run_meta.json` already logs it per run.

Don't collapse `condition` and `context.strategy` back into one field — that's what keeps both extensions restructuring-free.

## Backend: Ollama (default) vs. vLLM (selectable, `vllm-backend` branch)

`condition`/`context.strategy` select *what* gets run; the orthogonal `backend` config field (`"ollama"` default, or `"vllm"`) selects *which server* runs it. This exists because the remote box's NVIDIA driver (535.154.05, CUDA 12.2 ceiling) can't run conda-forge's `ollama` CUDA builds (only ship `cuda_129`/`cuda_130`), so a 36-hour CPU-only C2 run motivated pulling `meta-llama/Llama-3.1-8B-Instruct` straight from Hugging Face and serving it via vLLM instead, on the box's 2x idle Quadro RTX 5000 (16GB each).

- **Dispatcher, not a rewrite**: `call_ollama_once`/`call_ollama_with_retry`/`ensure_ollama_reachable` are completely untouched. `call_llm_once`/`call_llm_with_retry`/`ensure_backend_reachable` are new thin wrappers that branch on `cfg.get("backend", "ollama")`; the 4 call sites (`process_one`, `process_one_c2ab`, `select_stage1`, `process_one_c2c`) call the dispatcher instead of the Ollama functions directly, passing `cfg` through (all 4 already had `cfg` available). Any config without a `"backend"` key — i.e. all pre-existing configs — behaves byte-for-byte identically to before this existed.
- **vLLM runs as its own server** (`vllm serve ...`), same shape as `ollama serve`. `src/run.py` talks to it over HTTP via `requests` against the OpenAI-compatible `/v1/chat/completions` endpoint — it does **not** import the `vllm` package. vLLM's own (heavy, CUDA-pinned) dependencies live in a **separate** conda env (`environment-vllm.yml`, `venv-vllm`), kept out of the main `venv`/`requirements.txt` specifically so they can't conflict with the CPU-only `torch` already pinned there for C2b's embedding strategy.
- **`vllm==0.7.3` is pinned for Turing GPU support, not primarily for CUDA-version reasons.** The Quadro RTX 5000 is Turing (sm_75): no native bfloat16 (Llama 3.1's config ships bf16 by default) and no FlashAttention 2 (needs Ampere+). vLLM 0.7.3 falls back to the XFormers attention backend gracefully on Turing — **a startup warning about this is expected, not a failure** — while newer vLLM's V1 engine has progressively weaker Turing support. Its CUDA-12.1-default wheel (last release before a Feb 27 2025 PR switched the PyPI default to CUDA 12.4) happens to also satisfy the 12.2 driver ceiling, but that's a secondary bonus, not the reason for the pin: CUDA minor-version compatibility means `cu124` wheels generally run fine on driver ≥525.60.13 too — only a CUDA **13** wheel would be a true driver-version wall.
- **`vllm serve` invocation always needs three flags**, none optional:
  - `--dtype float16` — Llama 3.1's config defaults to bf16; Turing can't do bf16 and vLLM 0.7.3 errors at startup instead of auto-downgrading.
  - `--max-model-len 8192` (starting point; C1/C2's `k=4` context windows are short, this is generous headroom) — Llama 3.1 defaults to a 131072-token context, and vLLM pre-allocates KV cache sized for whatever `--max-model-len` is, which won't fit in the ~8GB/GPU left after model weights on `--tensor-parallel-size 2`. Tune with `--gpu-memory-utilization` if it still doesn't fit.
  - `--tensor-parallel-size 2` — **not falling back to `--tensor-parallel-size 1`** if this misbehaves (RTX 5000s have no NVLink, PCIe-only NCCL can be finicky): fp16 8B weights alone are ~16GB, i.e. one entire card's VRAM, before KV cache/activations/CUDA context. If TP=2 doesn't work, the real fallback is a **4-bit quantized checkpoint (AWQ/GPTQ) on a single GPU**, not TP=1 at full precision.
- **`guided_json` is the confirmed-correct structured-output param** for vLLM 0.7.x's OpenAI-compatible server (top-level request field, not nested under `extra_body` — that nesting is an `openai`-python-client-only concept, irrelevant to raw HTTP via `requests`). Kept isolated in `_guided_json_kwargs` regardless, so a future vLLM version needing a different shape (e.g. OpenAI-style `response_format={"type": "json_schema", ...}`) is a one-function fix.
- **Ollama's per-call `options.num_ctx`/`num_predict` don't map onto vLLM the same way.** `num_predict` (max output tokens) has a direct equivalent — `max_tokens` in the request body — but no current config sets it (audited all pre-vLLM configs: none set anything beyond `temperature`/`seed`), so `_call_vllm_once` doesn't send it yet; add the mapping if a config ever needs it. `num_ctx` (context window) has **no per-request equivalent at all** in vLLM's API — context length is fixed server-side once, at `vllm serve` startup via `--max-model-len`, not adjustable per-call the way Ollama's `options` dict might suggest.
- Gated HF model access (`meta-llama/Llama-3.1-8B-Instruct`) blocks using the official weights, but not environment/server verification — use the ungated mirror `NousResearch/Meta-Llama-3.1-8B-Instruct` (same weights) to unblock everything up through a working `vllm serve` + smoke completion while access clears.

## Invariants to preserve

- `src/run.py`'s inference loop must never raise on a bad/missing/malformed LLM response, for either backend. One retry (`call_ollama_with_retry`/`_call_vllm_with_retry`), then record whatever's salvageable (nulls where invalid) and continue. A single bad utterance must never kill a multi-hour remote run.
- `--dry-run` must stay completely network-free (no Ollama import/call, no `requests` call to a vLLM server either) — it's the fast local sanity check for prompt rendering. `ensure_backend_reachable` is called in `main()` only after the dry-run early return, for both backends.
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

# C2 (retrieval): random / sim / llm_select, val + test twins
python -m src.run --config configs/c2_llm_llama31_val.json --dry-run
python -m src.run --config configs/c2_random_llama31_val.json --smoke
python -m src.run --config configs/c2_sim_llama31_val.json --smoke
python -m src.run --config configs/c2_llm_llama31_val.json --smoke
python -m src.run --config configs/c2_random_llama31_val.json
python -m src.run --config configs/c2_sim_llama31_val.json
python -m src.run --config configs/c2_llm_llama31_val.json
python -m src.score --run outputs/c2_random_llama31_val
python -m src.score --run outputs/c2_sim_llama31_val
python -m src.score --run outputs/c2_llm_llama31_val
python -m src.score --run outputs/c2_llm_llama31_val --compare-selections outputs/c2_sim_llama31_val
python -m src.score --run outputs/c2_sim_llama31_val --compare-selections

# vLLM backend (vllm-backend branch): server runs in its own env, separately
conda activate venv-vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --dtype float16 --max-model-len 8192 --port 8000
# ... then, back in the main venv env:
conda activate venv
python -m src.run --config configs/c0_vllm_llama31_val.json --dry-run
python -m src.run --config configs/c0_vllm_llama31_val.json --smoke
python -m src.run --config configs/c0_vllm_llama31_val.json
python -m src.score --run outputs/c0_vllm_llama31_val
```

## Testing

`pytest tests/` needs no live Ollama server and no live vLLM server — `ollama.chat`/`ollama.list` are mocked where exercised (see `tests/test_run_resume.py` for the `unittest.mock.patch("ollama.chat", ...)` pattern to follow for any new Ollama-touching code; `tests/test_c2_llm.py` follows the same pattern for C2c's stage-1 selection call; `tests/test_run_vllm.py` mocks `requests.post`/`requests.get` the same way for the vLLM backend, and includes a test proving a `backend`-less config never touches `requests` at all). The split-parity test (`Session5` == 2195 rows) auto-skips if `data/iemocap/iemocap_merged_all.csv` isn't present. C2b's `_strategy_sim` tests monkeypatch `src.data._encode_batch` directly (see `tests/test_data.py`) so they need neither `torch`/`transformers` to be installed nor network access.
