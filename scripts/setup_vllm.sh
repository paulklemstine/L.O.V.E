#!/bin/bash
set -e

VENV_NAME=".venv_vllm"
PYTHON_EXEC="python3"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}== Setting up vLLM Environment ==${NC}"

# Check for Python
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_EXEC not found.${NC}"
    exit 1
fi

# Create venv if needed
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Creating virtual environment '$VENV_NAME'...${NC}"
    $PYTHON_EXEC -m venv "$VENV_NAME"
else
    echo -e "${GREEN}Virtual environment '$VENV_NAME' already exists.${NC}"
fi

# Activate venv
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install vLLM and dependencies.
# We hardcode a known good configuration or valid requirements just for vLLM to minimize conflicts.
echo -e "${YELLOW}Installing vLLM...${NC}"

# If requirements-vllm.txt exists, use it, otherwise install direct
if [ -f "requirements-vllm.txt" ]; then
    pip install -r requirements-vllm.txt
else
    # Fallback/Default install - using version from lovev1 findings (~0.11.1 or latest stable if not specified)
    # Note: Using a newer version might be better, but let's stick to something compatible.
    # Actually, let's try to install the latest consistent vllm unless user specified otherwise.
    # Given the user wants to "move that up", I should try to use the version they had or a robust one.
    # The lovev1 requirements said vllm>=0.11.1.
    pip install "vllm>=0.11.1" 
fi

echo -e "${GREEN}vLLM Setup Complete!${NC}"
