#!/bin/bash
set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as errors

# Define virtual environment path
VENV_ACTIVATE=".venv/bin/activate"

# Check if the virtual environment exists
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "[INFO] Virtuelle Umgebung nicht gefunden. Erstelle .venv..."
    python3 -m venv .venv || {
        echo "[ERROR] Konnte virtuelle Umgebung nicht erstellen."
        exit 1
    }
fi

# Activate virtual environment
source "$VENV_ACTIVATE" || {
    echo "[ERROR] Konnte virtuelle Umgebung nicht aktivieren."
    exit 1
}

# Upgrade pip
python3 -m pip install --upgrade pip || {
    echo "[ERROR] Konnte pip nicht aktualisieren."
    exit 1
}

# Install dependencies
pip install --disable-pip-version-check -r requirements.txt || {
    echo "[ERROR] Konnte Abhängigkeiten nicht installieren."
    exit 1
}

# Environment variables
export GPT4ALL_URL="http://localhost:4891/v1/chat/completions"
export MODEL_ID="DeepSeek-R1-Distill-Qwen-14B"
export ENGINE_MODE="elo"
export LLM_TIMEOUT="1800"

# Important: now controlled via max_tokens (server does not support n_predict)
export SEND_MAX_TOKENS="true"
export MAX_TOKENS="1500"
export SIMPLE_MAX_TOKENS="1500"

# Optional fine-tuning
export LLM_TEMPERATURE="0.2"
export LLM_TOP_P="0.95"
export LLM_REPEAT_PENALTY="1.1"

# Start the server
python3 -m uvicorn app:app --reload --port 8000
