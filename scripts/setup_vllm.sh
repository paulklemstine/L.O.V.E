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

# Install vLLM if missing
if ! pip show vllm &> /dev/null; then
    echo -e "${YELLOW}Installing vLLM...${NC}"
    if [ -f "requirements-vllm.txt" ]; then
        pip install -r requirements-vllm.txt
    else
        pip install "vllm>=0.15.0"
    fi
else
    echo -e "${GREEN}vLLM is already installed.${NC}"
fi

# Ensure bitsandbytes is installed (often missed if vLLM was pre-installed or system-wide)
if ! pip show bitsandbytes &> /dev/null; then
    echo -e "${YELLOW}Installing bitsandbytes...${NC}"
    pip install "bitsandbytes>=0.46.1"
else
    echo -e "${GREEN}bitsandbytes is already installed.${NC}"
fi

echo -e "${GREEN}vLLM Setup Complete!${NC}"
