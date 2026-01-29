#!/usr/bin/env bash
# setup_parallel.sh
#
# Setup + run IKIZAMINI parallel UI (Ollama) in one command.
# - Creates/uses venv/
# - Installs Python deps
# - Optionally starts Ollama and pulls models
# - Runs ikizamini_local_parallel.py (Flask UI with parallel objective processing)
#
# Defaults are Runpod-friendly (Ollama at localhost:11434, output to Parallely_Processed/).
#
# Usage:
#   ./setup_parallel.sh
#
# Optional environment variables:
#   (UI runs; uploads happen in browser. These vars mainly tune models/timeouts.)
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

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_err() { echo -e "${RED}[ERROR]${NC} $*"; }

SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    log_err "Not running as root and sudo not found; cannot install system packages."
    exit 1
  fi
fi

PKG_MANAGER=""
INSTALL_CMD=""
UPDATE_CMD=""

detect_package_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER="apt-get"
    UPDATE_CMD="$SUDO apt-get update -y"
    INSTALL_CMD="$SUDO apt-get install -y"
  elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    UPDATE_CMD="$SUDO dnf makecache -y"
    INSTALL_CMD="$SUDO dnf install -y"
  elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    UPDATE_CMD="$SUDO yum makecache -y"
    INSTALL_CMD="$SUDO yum install -y"
  elif command -v pacman >/dev/null 2>&1; then
    PKG_MANAGER="pacman"
    UPDATE_CMD="$SUDO pacman -Sy --noconfirm"
    INSTALL_CMD="$SUDO pacman -S --noconfirm"
  else
    log_err "Could not detect a supported package manager (apt-get/dnf/yum/pacman)."
    exit 1
  fi
}

install_system_deps() {
  detect_package_manager
  log_info "Updating package lists ($PKG_MANAGER)..."
  # shellcheck disable=SC2086
  $UPDATE_CMD >/dev/null 2>&1 || true

  # Minimum packages needed for Ollama install + building venv reliably
  log_info "Installing system packages (curl, ca-certificates, zstd, python venv tooling)..."
  if [ "$PKG_MANAGER" = "apt-get" ]; then
    # shellcheck disable=SC2086
    $INSTALL_CMD curl ca-certificates zstd python3-venv python3-pip >/dev/null
  else
    # Best-effort across distros
    # shellcheck disable=SC2086
    $INSTALL_CMD curl ca-certificates zstd python3 python3-pip >/dev/null || true
  fi

  if ! command -v curl >/dev/null 2>&1; then
    log_err "curl is still missing after install. Please install curl manually."
    exit 1
  fi
}

install_ollama_if_needed() {
  if command -v ollama >/dev/null 2>&1; then
    log_ok "Ollama already installed: $(ollama --version 2>/dev/null || echo installed)"
    return
  fi

  log_info "Installing Ollama..."
  # Official install script; on containers systemd warnings are normal.
  curl -fsSL https://ollama.com/install.sh | sh

  if ! command -v ollama >/dev/null 2>&1; then
    log_err "Ollama installation failed."
    exit 1
  fi
  log_ok "Ollama installed: $(ollama --version 2>/dev/null || echo installed)"
}

IK_OLLAMA_URL="${IK_OLLAMA_URL:-http://localhost:11434}"
IK_WORKER_MODEL="${IK_WORKER_MODEL:-qwen:32b}"
IK_MANAGER_MODEL="${IK_MANAGER_MODEL:-qwen:32b}"
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
echo "Ollama URL:  $IK_OLLAMA_URL"
echo "Models:      worker=$IK_WORKER_MODEL manager=$IK_MANAGER_MODEL"
echo "Objective concurrency: $IK_WORKERS"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 1
fi

install_system_deps
install_ollama_if_needed

if [ ! -d "venv" ]; then
  log_info "Creating venv/ ..."
  python3 -m venv venv
fi

log_info "Activating venv and installing Python deps ..."
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q flask requests jsonschema openai

START_FLAGS=()
if [ "$IK_START_OLLAMA" = "1" ]; then
  # start Ollama server if needed
  : # handled below
fi

if [ "$IK_START_OLLAMA" = "1" ]; then
  log_info "Ensuring Ollama is running ..."
  # Start in background if not already responding (non-fatal if already running)
  if ! curl -fsS "$IK_OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    ollama serve >/dev/null 2>&1 &
    # Wait until ready (max IK_WAIT_OLLAMA seconds)
    for i in $(seq 1 "$IK_WAIT_OLLAMA"); do
      if curl -fsS "$IK_OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if ! curl -fsS "$IK_OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      log_err "Ollama did not become ready at $IK_OLLAMA_URL after ${IK_WAIT_OLLAMA}s"
      exit 1
    fi
  fi
fi

if [ "$IK_PULL_MODEL" = "1" ]; then
  log_info "Ensuring models are available (ollama pull if missing) ..."
  ollama pull "$IK_WORKER_MODEL" >/dev/null
  if [ "$IK_MANAGER_MODEL" != "$IK_WORKER_MODEL" ]; then
    ollama pull "$IK_MANAGER_MODEL" >/dev/null
  fi
fi

export IKIZAMINI_PARALLEL_WORKERS="$IK_WORKERS"
export IKIZAMINI_OLLAMA_TIMEOUT="$IK_TIMEOUT_S"
export IKIZAMINI_OLLAMA_RETRIES="$IK_MAX_RETRIES"
export IKIZAMINI_DEFAULT_WORKER_MODEL="$IK_WORKER_MODEL"
export IKIZAMINI_DEFAULT_MANAGER_MODEL="$IK_MANAGER_MODEL"

log_info "Starting IKIZAMINI Parallel UI ..."
python3 -u ikizamini_local_parallel.py

