#!/bin/bash

# J.U.L.E.S. Termux Bootstrapper
# This script is the single entry point for setting up and running the application on Termux.

echo "========================================="
echo " J.U.L.E.S. Termux Environment Setup"
echo "========================================="
echo "This script will install all necessary packages."
echo "It may take a significant amount of time, especially"
echo "during the compilation of the llama.cpp library."
echo "Please ensure your device is connected to a stable"
echo "Wi-Fi network and has sufficient battery."
echo "========================================="
# The script is non-interactive from here.

# --- Update and Upgrade Termux Packages ---
echo ""
echo "[*] Updating Termux package lists and upgrading existing packages..."
pkg update -y && pkg upgrade -y
if [ $? -ne 0 ]; then
    echo "[!] Failed to update Termux packages. Please check your internet connection."
    exit 1
fi
echo "[+] Termux packages updated successfully."

# --- Install Core System Dependencies ---
echo ""
echo "[*] Installing core system dependencies (python, nodejs-lts, git, clang)..."
pkg install -y python nodejs-lts git clang make libjpeg-turbo libcrypt
if [ $? -ne 0 ]; then
    echo "[!] Failed to install core system packages."
    exit 1
fi
echo "[+] Core system dependencies installed successfully."

# --- Install Node.js Dependencies for P2P Bridge ---
if [ -f "package.json" ]; then
    echo ""
    echo "[*] Installing Node.js dependencies from package.json..."
    npm install
    if [ $? -ne 0 ]; then
        echo "[!] Failed to install Node.js dependencies."
        exit 1
    fi
    echo "[+] Node.js dependencies installed successfully."
else
    echo "[!] package.json not found. Skipping npm install."
fi

# --- Delegate to evolve.py for Python Dependencies ---
echo ""
echo "[*] Handing off to evolve.py for platform-aware Python dependency installation..."
echo "[*] This is the longest step and will involve compiling code. Please be patient."
# We run the script with --help. The dependency check runs before arg parsing.
# This avoids starting the full application while still triggering the setup.
python evolve.py --help > /dev/null
if [ $? -ne 0 ]; then
    echo "[!] The dependency installer in evolve.py failed."
    echo "[!] Please check the output above for errors."
    exit 1
fi
echo "[+] All Python dependencies installed successfully."

# --- Launch J.U.L.E.S. ---
echo ""
echo "========================================="
echo "  SETUP COMPLETE. LAUNCHING J.U.L.E.S."
echo "========================================="
echo ""

python evolve.py

# --- Post-Execution ---
echo ""
echo "========================================="
echo " J.U.L.E.S. has terminated."
echo "========================================="
read -p "Press [Enter] to exit the terminal."