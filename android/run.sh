#!/bin/bash

# J.U.L.E.S. Android Launcher
# This script starts the main evolve.py application.

# Navigate to the root of the repository
cd "$(dirname "$0")/.."

# Check if the state file exists, if not, prompt the user
if [ ! -f "jules_state.json" ]; then
    echo "It looks like this is the first time you are running J.U.L.E.S."
    echo "A 'jules_state.json' file will be created to store the AI's memory."
    echo ""
fi

# Launch the main Python script
python evolve.py