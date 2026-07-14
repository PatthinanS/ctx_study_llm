#!/usr/bin/env bash
# Self-contained bootstrap for running ContextStudy_LLM on a remote box with
# NO ROOT ACCESS. Idempotent: safe to re-run.
#
# Both the Ollama server binary and the Python dependencies come from a
# single conda env (default name "venv", see environment.yml): the
# conda-forge "ollama" package for the CLI/server, plus python + pip deps.
# Using conda-forge's build (rather than Ollama's own installer/binary,
# which needs a fairly recent glibc -- e.g. requires GLIBC_2.28+) also
# sidesteps old-glibc remote boxes, since conda-forge targets a much older
# baseline glibc for portability. The exact ollama version+build is pinned
# in environment.yml -- newer conda-forge builds (0.30.x) ship without the
# llama-server runner binary (see CLAUDE.md), so don't unpin this casually.
#
# Requires `conda` (e.g. Miniconda) already installed -- Miniconda itself
# installs into $HOME, no root needed, but this script does not install
# conda itself.
set -euo pipefail

cd "$(dirname "$0")"

OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b-instruct-q4_K_M}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-venv}"

echo "== 1/4: Conda environment (${CONDA_ENV_NAME}), incl. Ollama + Python deps =="
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    echo "Install Miniconda (no root needed, installs into \$HOME):" >&2
    echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
    echo "Updating existing conda env '${CONDA_ENV_NAME}'..."
    conda env update -n "${CONDA_ENV_NAME}" -f environment.yml --prune
else
    echo "Creating conda env '${CONDA_ENV_NAME}'..."
    conda env create -n "${CONDA_ENV_NAME}" -f environment.yml
fi
conda activate "${CONDA_ENV_NAME}"

command -v ollama >/dev/null 2>&1 || {
    echo "ERROR: ollama not on PATH after conda env activation -- check environment.yml." >&2
    exit 1
}
echo "ollama: $(command -v ollama) ($(ollama --version 2>&1 | head -1))"

mkdir -p outputs

echo "== 2/4: Ollama server =="
server_up() {
    curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null 2>&1
}

if server_up; then
    echo "Ollama server already reachable at ${OLLAMA_HOST_URL}"
else
    echo "Starting Ollama server in background (log: outputs/ollama_serve.log)..."
    nohup ollama serve > outputs/ollama_serve.log 2>&1 &
    disown

    waited=0
    until server_up; do
        if [ "$waited" -ge 30 ]; then
            echo "ERROR: Ollama server did not become reachable within 30s." >&2
            echo "Check outputs/ollama_serve.log for details." >&2
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "Ollama server is up at ${OLLAMA_HOST_URL}"
fi

echo "== 3/4: Pull default model (${OLLAMA_MODEL}) =="
ollama pull "${OLLAMA_MODEL}"

echo "== 4/4: Data directory =="
mkdir -p data/iemocap

cat <<EOF

Setup complete.

  ollama binary: $(command -v ollama)
  conda env:     ${CONDA_ENV_NAME}
  ollama server: ${OLLAMA_HOST_URL} (log: outputs/ollama_serve.log)
  model pulled:  ${OLLAMA_MODEL}

Next steps:
  conda activate ${CONDA_ENV_NAME}
  # copy iemocap_merged_all.csv into data/iemocap/ if not already present

  python -m src.run --config configs/c0_mistral.json --dry-run
  python -m src.run --config configs/c1_mistral.json --dry-run
  python -m src.run --config configs/c0_mistral.json --smoke
  python -m src.run --config configs/c1_mistral.json --smoke
  python -m src.run --config configs/c0_mistral.json
  python -m src.run --config configs/c1_mistral.json
  python -m src.score --run outputs/c0_mistral
  python -m src.score --run outputs/c1_mistral
EOF
