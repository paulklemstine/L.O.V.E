#!/bin/bash

# Fix vLLM and PyTorch dependency mismatch
# This script uninstalls current versions and installs known compatible ones.

echo "Uninstalling vllm, torch, torchvision, torchaudio, xformers..."
python3 -m pip uninstall -y vllm torch torchvision torchaudio xformers --break-system-packages

echo "Installing compatible vllm and torch..."
# vLLM 0.6.4.post1 matches usually with Torch 2.5.1
# We let pip resolve the dependencies by installing vllm first.
python3 -m pip install "vllm>=0.6.4" "torch==2.5.1" "torchvision" "torchaudio" --extra-index-url https://download.pytorch.org/whl/cu124 --break-system-packages

echo "Verifying installation..."
python3 -c "import torch; print(f'Torch version: {torch.__version__}'); import vllm; print(f'vLLM version: {vllm.__version__}')"

echo "Done."
