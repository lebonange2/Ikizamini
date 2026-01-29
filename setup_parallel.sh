#!/usr/bin/env bash
# setup_parallel.sh
#
# Setup + run IKIZAMINI parallel runner (Ollama) in one command.
# - Creates/uses venv/
# - Installs Python deps
# - Optionally starts Ollama and pulls models
# - Runs ikizamini_local_parallel.py
#
# Defaults are Runpod-friendly (Ollama at localhost:11434, output to Parallely_Processed/).
#
# Usage:
#   ./setup_parallel.sh
#
# Optional environment variables:
#   IK_INPUT="Uru.txt"
#   IK_OUTPUT_DIR="Parallely_Processed"
#   IK_OLLAMA_URL="http://localhost:11434"
#   IK_WORKER_MODEL="gemma3:latest"
#   IK_MANAGER_MODEL="gemma3:latest"
#   IK_WORKERS="4"
#   IK_LIMIT="0"
#   IK_MAX_ROUNDS="6"
#   IK_NUM_CTX="8192"
#   IK_TIMEOUT_S="600"
#   IK_MAX_RETRIES="4"
#   IK_PULL_MODEL="1"         # 1 to pull models if missing, 0 to skip
#   IK_START_OLLAMA="1"       # 1 to start `ollama serve` if not responding, 0 to skip
#   IK_WAIT_OLLAMA="60"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IK_INPUT="${IK_INPUT:-Uru.txt}"
IK_OUTPUT_DIR="${IK_OUTPUT_DIR:-Parallely_Processed}"
IK_OLLAMA_URL="${IK_OLLAMA_URL:-http://localhost:11434}"
IK_WORKER_MODEL="${IK_WORKER_MODEL:-gemma3:latest}"
IK_MANAGER_MODEL="${IK_MANAGER_MODEL:-gemma3:latest}"
IK_WORKERS="${IK_WORKERS:-4}"
IK_LIMIT="${IK_LIMIT:-0}"
IK_MAX_ROUNDS="${IK_MAX_ROUNDS:-6}"
IK_NUM_CTX="${IK_NUM_CTX:-8192}"
IK_TIMEOUT_S="${IK_TIMEOUT_S:-600}"
IK_MAX_RETRIES="${IK_MAX_RETRIES:-4}"
IK_PULL_MODEL="${IK_PULL_MODEL:-1}"
IK_START_OLLAMA="${IK_START_OLLAMA:-1}"
IK_WAIT_OLLAMA="${IK_WAIT_OLLAMA:-60}"

echo "=========================================="
echo "IKIZAMINI Parallel Setup + Run"
echo "=========================================="
echo "Input:       $IK_INPUT"
echo "Output dir:  $IK_OUTPUT_DIR"
echo "Ollama URL:  $IK_OLLAMA_URL"
echo "Models:      worker=$IK_WORKER_MODEL manager=$IK_MANAGER_MODEL"
echo "Concurrency: $IK_WORKERS"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "[SETUP] Creating venv/ ..."
  python3 -m venv venv
fi

echo "[SETUP] Activating venv and installing deps ..."
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q flask requests jsonschema openai

if ! command -v ollama >/dev/null 2>&1; then
  echo "[ERROR] ollama not found. Install Ollama first (or run ./setup.sh)."
  exit 1
fi

START_FLAGS=()
if [ "$IK_START_OLLAMA" = "1" ]; then
  START_FLAGS+=(--start-ollama --wait-ollama "$IK_WAIT_OLLAMA")
fi

PULL_FLAGS=()
if [ "$IK_PULL_MODEL" = "1" ]; then
  PULL_FLAGS+=(--pull-model)
fi

LIMIT_FLAGS=()
if [ "${IK_LIMIT}" != "0" ]; then
  LIMIT_FLAGS+=(--limit "$IK_LIMIT")
fi

echo "[RUN] Starting parallel runner ..."
python3 -u ikizamini_local_parallel.py \
  --input "$IK_INPUT" \
  --output-dir "$IK_OUTPUT_DIR" \
  --ollama-url "$IK_OLLAMA_URL" \
  --worker-model "$IK_WORKER_MODEL" \
  --manager-model "$IK_MANAGER_MODEL" \
  --workers "$IK_WORKERS" \
  --max-rounds "$IK_MAX_ROUNDS" \
  --num-ctx "$IK_NUM_CTX" \
  --timeout-s "$IK_TIMEOUT_S" \
  --max-retries "$IK_MAX_RETRIES" \
  "${START_FLAGS[@]}" \
  "${PULL_FLAGS[@]}" \
  "${LIMIT_FLAGS[@]}"

