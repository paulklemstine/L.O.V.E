#!/bin/bash
set -e

# DEBUG: Print args
echo "DEBUG START_VLLM: Received args: $@"

# Default settings
HOST="0.0.0.0"
PORT="8000"
MODEL="" # If empty, vLLM will default or we pass it via args

# Parse arguments to extract --venv
# We need to preserve other args for vLLM
ARGS=()
VENV_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore args for vLLM
set -- "${ARGS[@]}"

# Fallback path logic
if [ -z "$VENV_PATH" ]; then
    if [ -n "$VLLM_VENV_PATH" ]; then
        VENV_PATH="$VLLM_VENV_PATH"
    else
        VENV_PATH="$(dirname "$0")/../.venv_vllm"
    fi
fi

# Activate venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: vLLM virtual environment not found at $VENV_PATH"
    echo "Please run scripts/setup_vllm.sh first."
    exit 1
fi

echo "Starting vLLM..."
echo "Using venv: $VENV_PATH"
# We use exec so the shell process is replaced by vLLM
exec python3 -m vllm.entrypoints.openai.api_server --host "$HOST" --port "$PORT" "$@"
