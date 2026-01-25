#!/bin/bash
set -e

# Default settings
HOST="0.0.0.0"
PORT="8000"
MODEL="" # If empty, vLLM will default or we pass it via args
VENV_PATH="$(dirname "$0")/../.venv_vllm"

# Activate venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: vLLM virtual environment not found at $VENV_PATH"
    echo "Please run scripts/setup_vllm.sh first."
    exit 1
fi

echo "Starting vLLM..."
# We use exec so the shell process is replaced by vLLM, easier for signal handling (killing this script kills vLLM)
# Pass all arguments to the underlying command
exec python3 -m vllm.entrypoints.openai.api_server --host "$HOST" --port "$PORT" "$@"
