#!/bin/bash

# Ensure environments are set up
echo "Initializing environments..."
bash setup_envs.sh

# Activate Core Environment
source .venv_core/bin/activate

while true; do
    # Run L.O.V.E. with unbuffered output
    python -u love.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 42 ]; then
        echo "Hot Restart triggered. Reloading L.O.V.E..."
        continue
    else
        break
    fi
done
