#!/bin/bash
set -e
# Try to install python3-venv if we have apt-get and sudo (fixes some Colab/Ubuntu envs)
if command -v apt-get &> /dev/null && command -v sudo &> /dev/null; then
  echo -e "${YELLOW}Checking for python3-venv...${NC}"
  sudo apt-get update && sudo apt-get install -y python3-venv || echo -e "${YELLOW}Could not install python3-venv, hoping it is already there.${NC}"
fi

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
    # Check if a critical file exists to confirm it's a valid venv
    if [ ! -f "$env_name/bin/python" ]; then
        echo -e "${YELLOW}Creating (or recreating) virtual environment: $env_name${NC}"
        rm -rf "$env_name" # Wipe potential broken dir
        python3 -m venv "$env_name"
        
        # Verify creation succeeded
        if [ ! -f "$env_name/bin/python" ]; then
             echo -e "${RED}ERROR: Failed to create venv at $env_name. Python binary missing.${NC}"
             exit 1
        fi
        
    else
        echo -e "${GREEN}Virtual environment $env_name appears valid (Python binary found).${NC}"
    fi

    # ALWAYS check for pip, regardless of whether we just created it or it existed
    # Some older venvs or --without-pip creations might lack it.
    if ! "$env_name/bin/python" -m pip --version &> /dev/null; then
        echo -e "${YELLOW}pip not found or working in $env_name. Bootstrapping...${NC}"
        "$env_name/bin/python" -m ensurepip --upgrade || {
            echo -e "${YELLOW}ensurepip failed. Trying manual get-pip.py fallback...${NC}"
            # Fallback for systems where ensurepip is broken/missing
            curl -sS https://bootstrap.pypa.io/get-pip.py | "$env_name/bin/python" || {
               echo -e "${RED}Critical Error: Could not install pip in venv.${NC}"
               exit 1
            }
        }
    fi
}

# --- Setup Core Environment ---
create_venv "$VENV_CORE"

echo -e "${YELLOW}Installing/Upgrading dependencies for Core App ($VENV_CORE)...${NC}"
# Upgrade pip in core env
# Upgrade pip in core env (use python -m pip to avoid shebang issues)
"$VENV_CORE/bin/python" -m pip install --upgrade pip

# Install core requirements
if [ -f "requirements.txt" ]; then
    "$VENV_CORE/bin/python" -m pip install -r "requirements.txt" -v || {
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
"$VENV_VLLM/bin/python" -m pip install --upgrade pip

# Install vLLM requirements
if [ -f "requirements-deepagent.txt" ]; then
    # Note: we don't install requirements.txt here, only deepagent specific stuff.
    # But deepagent might need some core stuff. Usually vllm pulls in what it needs.
    "$VENV_VLLM/bin/python" -m pip install -r "requirements-deepagent.txt" -v || {
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
