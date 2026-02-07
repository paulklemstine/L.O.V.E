#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1
node "/home/raver1975/L.O.V.E/external/pi-agent/packages/coding-agent/dist/cli.js" --mode rpc --extension "/home/raver1975/L.O.V.E/external/vllm-extension" --provider vllm --model kaitchup/Phi-3-mini-4k-instruct-gptq-4bit
