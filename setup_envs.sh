#!/bin/bash
set -e

# Define environment paths
VENV_CORE=".venv_core"
VENV_VLLM=".venv_vllm"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Dependency Isolation Setup...${NC}"

# Function to create venv if it doesn't exist
create_venv() {
    local env_name=$1
    if [ ! -d "$env_name" ]; then
        echo -e "${YELLOW}Creating virtual environment: $env_name${NC}"
        python3 -m venv "$env_name"
    else
        echo -e "${GREEN}Virtual environment $env_name already exists.${NC}"
    fi
}

# --- Setup Core Environment ---
create_venv "$VENV_CORE"

echo -e "${YELLOW}Installing/Upgrading dependencies for Core App ($VENV_CORE)...${NC}"
# Upgrade pip in core env
"$VENV_CORE/bin/pip" install --upgrade pip

# Install core requirements
if [ -f "requirements.txt" ]; then
    "$VENV_CORE/bin/pip" install -r "requirements.txt" || {
         echo -e "${RED}Failed to install core requirements. Attempting fix...${NC}"
         # Setup a fallback or retry here? For now, fail hard to alert user.
         exit 1
    }
else
    echo -e "${RED}requirements.txt not found!${NC}"
    exit 1
fi


# --- Setup vLLM Environment ---
create_venv "$VENV_VLLM"

echo -e "${YELLOW}Installing/Upgrading dependencies for vLLM ($VENV_VLLM)...${NC}"
# Upgrade pip in vllm env
"$VENV_VLLM/bin/pip" install --upgrade pip

# Install vLLM requirements
if [ -f "requirements-deepagent.txt" ]; then
    # Note: we don't install requirements.txt here, only deepagent specific stuff.
    # But deepagent might need some core stuff. Usually vllm pulls in what it needs.
    "$VENV_VLLM/bin/pip" install -r "requirements-deepagent.txt" || {
        echo -e "${RED}Failed to install vLLM requirements.${NC}"
        exit 1
    }
else
    echo -e "${RED}requirements-deepagent.txt not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}Core Environment: $VENV_CORE${NC}"
echo -e "${GREEN}vLLM Environment: $VENV_VLLM${NC}"
