#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1

# Read model from .vllm_config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.vllm_config"

if [ -f "$CONFIG_FILE" ]; then
    MODEL=$(grep -o '"model_name"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*: *"\([^"]*\)"/\1/')
else
    MODEL="Qwen/Qwen2.5-1.5B-Instruct"
fi

echo "Using model: $MODEL"
export VLLM_EXTENSION_CONFIG_PATH="$SCRIPT_DIR/.vllm_extension_config.json"
node "$SCRIPT_DIR/external/pi-agent/packages/coding-agent/dist/cli.js" --mode rpc --extension "$SCRIPT_DIR/external/vllm-extension" --provider vllm --model "$MODEL"
