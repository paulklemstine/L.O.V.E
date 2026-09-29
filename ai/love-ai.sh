#!/bin/bash
# love-ai.sh — local inference entry point for the L.O.V.E pipeline
# Usage:
#   love-ai.sh text   "<system prompt>" "<user prompt>"
#   love-ai.sh image  --prompt "..." [--out out.png] [--w 1024] [--h 1024] ...
set -e
# This script lives in the repo next to the image scripts it calls, so resolve
# them from here rather than assuming a checkout location.
AI_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/ai/ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/ai/ollama-models"
# Only the weights and the venv live outside the repo. Override with
# LOVE_AI_HOME if the toolchain moves.
export LOVE_AI_HOME="${LOVE_AI_HOME:-$HOME/ai}"

if ! python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:11434/api/version',timeout=2)" 2>/dev/null; then
  echo "starting ollama server..." >&2
  setsid "$LOVE_AI_HOME/start-ollama.sh" >/dev/null 2>&1 &
  for i in $(seq 30); do
    python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:11434/api/version',timeout=2)" 2>/dev/null && break
    sleep 1
  done
fi

case "$1" in
  text)
    python3 - "$2" "$3" <<'EOF'
import json, sys, urllib.request
system, user = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": "qwen3:8b",
    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    "temperature": 0.85, "stream": False,
}).encode()
req = urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"])
EOF
    ;;
  image)
    shift
    ollama stop qwen3:8b >/dev/null 2>&1 || true
    sleep 2
    exec "$LOVE_AI_HOME/imgenv/bin/python" "$AI_SCRIPTS/generate_image.py" "$@"
    ;;
  *)
    echo "usage: love-ai.sh {text|image} ..." >&2; exit 1 ;;
esac
