#!/usr/bin/env bash
# Self-contained bootstrap for running ContextStudy_LLM on a remote box.
# Idempotent: safe to re-run. Installs the Ollama server binary (not just
# the Python client), starts it, pulls the default model, and creates a venv
# with all Python dependencies.
set -euo pipefail

cd "$(dirname "$0")"

OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b-instruct-q4_K_M}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"

echo "== 1/5: Ollama server binary =="
if ! command -v ollama >/dev/null 2>&1; then
    os="$(uname -s)"
    if [ "$os" = "Linux" ]; then
        echo "Installing Ollama (Linux install script)..."
        curl -fsSL https://ollama.com/install.sh | sh
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
else
    echo "ollama already installed: $(command -v ollama)"
fi

echo "== 2/5: Python venv =="
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 3/5: Python dependencies =="
pip install --upgrade pip -q
pip install -r requirements.txt -q

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

  venv:          $(pwd)/.venv
  ollama server: ${OLLAMA_HOST_URL} (log: outputs/ollama_serve.log)
  model pulled:  ${OLLAMA_MODEL}

Next steps:
  source .venv/bin/activate
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
