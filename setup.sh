#!/usr/bin/env bash
# Self-contained bootstrap for running ContextStudy_LLM on a remote box with
# NO ROOT ACCESS. Idempotent: safe to re-run.
#
#   1. Ollama server binary: installed by extracting the official release
#      tarball into a user-writable prefix ($OLLAMA_INSTALL_DIR, default
#      ~/.local/ollama) instead of /usr/local -- this is exactly what
#      ollama's own install.sh does internally, minus the sudo-only steps
#      (root-owned /usr/local, a systemd service running as a dedicated
#      "ollama" user). None of that is required to just run `ollama serve`
#      as the current user.
#   2. Python environment: a conda env (default name "llm_leg") instead of
#      venv, created from environment.yml. Requires `conda` (e.g. Miniconda)
#      already installed -- Miniconda itself installs into $HOME, no root
#      needed, but this script does not install conda itself.
set -euo pipefail

cd "$(dirname "$0")"

OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b-instruct-q4_K_M}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
OLLAMA_INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$HOME/.local/ollama}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llm_leg}"

echo "== 1/5: Ollama server binary (no root required) =="
if command -v ollama >/dev/null 2>&1; then
    echo "ollama already on PATH: $(command -v ollama)"
elif [ -x "$OLLAMA_INSTALL_DIR/bin/ollama" ]; then
    echo "ollama already installed at $OLLAMA_INSTALL_DIR/bin/ollama"
    export PATH="$OLLAMA_INSTALL_DIR/bin:$PATH"
else
    os="$(uname -s)"
    if [ "$os" = "Linux" ]; then
        arch="$(uname -m)"
        case "$arch" in
            x86_64) arch="amd64" ;;
            aarch64|arm64) arch="arm64" ;;
            *) echo "ERROR: unsupported architecture '$arch'." >&2; exit 1 ;;
        esac
        echo "Downloading ollama-linux-${arch}.tgz into $OLLAMA_INSTALL_DIR (no sudo)..."
        mkdir -p "$OLLAMA_INSTALL_DIR"
        curl --fail --show-error --location --progress-bar \
            "https://ollama.com/download/ollama-linux-${arch}.tgz" | \
            tar -xzf - -C "$OLLAMA_INSTALL_DIR"
        export PATH="$OLLAMA_INSTALL_DIR/bin:$PATH"
        echo "NOTE: add this to your shell rc (~/.bashrc) to persist across sessions:"
        echo "  export PATH=\"$OLLAMA_INSTALL_DIR/bin:\$PATH\""
    elif [ "$os" = "Darwin" ]; then
        if command -v brew >/dev/null 2>&1; then
            echo "Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo "ERROR: ollama not found and Homebrew is not available." >&2
            echo "Install manually from https://ollama.com/download and re-run this script." >&2
            exit 1
        fi
    else
        echo "ERROR: unsupported OS '$os'. Install Ollama manually from https://ollama.com/download." >&2
        exit 1
    fi
fi

echo "== 2/5: Conda environment (${CONDA_ENV_NAME}) =="
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

echo "== 3/5: Python dependencies =="
echo "installed via environment.yml (conda env create/update above already ran pip install -r requirements.txt)"

mkdir -p outputs

echo "== 4/5: Ollama server =="
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

echo "== 5/5: Pull default model (${OLLAMA_MODEL}) =="
ollama pull "${OLLAMA_MODEL}"

cat <<EOF

Setup complete.

  ollama binary: $(command -v ollama)
  conda env:     ${CONDA_ENV_NAME}
  ollama server: ${OLLAMA_HOST_URL} (log: outputs/ollama_serve.log)
  model pulled:  ${OLLAMA_MODEL}

Next steps:
  conda activate ${CONDA_ENV_NAME}
  mkdir -p data/iemocap   # copy iemocap_merged_all.csv here if not already present

  python -m src.run --config configs/c0_mistral.json --dry-run
  python -m src.run --config configs/c1_mistral.json --dry-run
  python -m src.run --config configs/c0_mistral.json --smoke
  python -m src.run --config configs/c1_mistral.json --smoke
  python -m src.run --config configs/c0_mistral.json
  python -m src.run --config configs/c1_mistral.json
  python -m src.score --run outputs/c0_mistral
  python -m src.score --run outputs/c1_mistral
EOF
