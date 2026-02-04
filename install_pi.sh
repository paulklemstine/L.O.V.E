#!/bin/bash
set -e

# Define project root
PROJECT_ROOT="$HOME/L.O.V.E"
AGENT_DIR="$PROJECT_ROOT/external/pi-agent"

echo ">>> Starting Pi Agent Setup in WSL..."

# 1. Install NVM if not present
if [ -z "$NVM_DIR" ]; then
    export NVM_DIR="$HOME/.nvm"
fi

if [ ! -d "$NVM_DIR" ]; then
    echo ">>> Installing NVM..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
else
    echo ">>> NVM directory found."
fi

# 2. Load NVM
echo ">>> Loading NVM..."
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 3. Install Node 22
echo ">>> Installing Node 22..."
nvm install 22
nvm use 22

# Verify Node version
NODE_LOC=$(which node)
echo ">>> Using Node at: $NODE_LOC"
node -v

# 4. Install Dependencies and Build
echo ">>> Building Pi Agent..."
cd "$AGENT_DIR" || exit 1

# Ensure we are not using Windows npm
if [[ "$(which npm)" == *"/mnt/c/"* ]]; then
    echo "ERROR: Still picking up Windows npm. NVM setup failed to shadow it."
    exit 1
fi

echo ">>> Running npm install..."
npm install

echo ">>> Running npm run build..."
npm run build

echo ">>> SUCCESS: Pi Agent built successfully."
