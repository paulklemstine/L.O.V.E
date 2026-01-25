#!/bin/bash

# Build vLLM from source
# USE THIS ONLY IF the binary fix (fix_vllm_dependencies.sh) fails.
# This ensures vLLM is compiled against the EXACT version of PyTorch you have installed.

echo "Installing build dependencies..."
python3 -m pip install cmake packaging wheel ninja --break-system-packages

echo "Cloning vLLM repository..."
git clone https://github.com/vllm-project/vllm.git
cd vllm

echo "Building and installing vLLM (this may take 10-20 minutes)..."
# We use --no-build-isolation to use the system installed PyTorch
python3 -m pip install . --break-system-packages --no-build-isolation

echo "Verifying installation..."
cd ..
python3 -c "import vllm; print(f'Success! vLLM version: {vllm.__version__}')"
