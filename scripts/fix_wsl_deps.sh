#!/bin/bash
# Fix WSL dependencies for L.O.V.E. v2

echo "🔧 Installing system build dependencies (requires sudo)..."
echo "You may be asked for your password."
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

echo "📦 Upgrading pip and build tools..."
python3 -m pip install --upgrade pip setuptools wheel --break-system-packages

echo "🚀 Retrying dependency installation..."
python3 -m pip install -r requirements.txt --break-system-packages

echo "✅ Done! You can now run 'python3 run.py'."
