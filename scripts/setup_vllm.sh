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

# Ensure python3-venv is installed (common issue on Debian/Ubuntu/Colab)
if command -v apt-get &> /dev/null; then
    if ! dpkg -s python3-venv &> /dev/null; then
        echo -e "${YELLOW}Installing python3-venv...${NC}"
        # Try with sudo, fallback to ignore if not available/needed
        sudo apt-get update && sudo apt-get install -y python3-venv || echo -e "${YELLOW}Apt install failed (might not have sudo), hoping venv works...${NC}"
    fi
fi

# Create venv if needed
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Creating virtual environment '$VENV_NAME'...${NC}"
    
    # Try standard creation first
    if ! $PYTHON_EXEC -m venv "$VENV_NAME"; then
        echo -e "${YELLOW}Standard venv creation failed (likely ensurepip). Retrying without pip...${NC}"
        # Fallback: Create without pip, then bootstrap it
        if $PYTHON_EXEC -m venv --without-pip "$VENV_NAME"; then
            echo -e "${YELLOW}Bootstrapping pip manually...${NC}"
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
            "$VENV_NAME/bin/python3" get-pip.py
            rm get-pip.py
        else
            echo -e "${RED}CRITICAL: Failed to create virtual environment.${NC}"
            exit 1
        fi
    fi
    
    if [ ! -f "$VENV_NAME/bin/activate" ]; then
        echo -e "${RED}CRITICAL: Failed to create virtual environment. 'bin/activate' is missing.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Virtual environment '$VENV_NAME' already exists.${NC}"
fi

# Activate venv
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Check if vLLM is already installed
if pip show vllm &> /dev/null; then
    echo -e "${GREEN}vLLM is already installed. Skipping installation.${NC}"
else
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
fi

echo -e "${GREEN}vLLM Setup Complete!${NC}"
