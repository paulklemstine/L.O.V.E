#!/bin/bash

# J.U.L.E.S. Android Setup Script
# This script prepares the Termux environment for running evolve.py.

echo "========================================="
echo " J.U.L.E.S. Termux Environment Setup"
echo "========================================="
echo "This script will install the necessary system packages and then"
echo "delegate the Python-specific setup to the main evolve.py script."
echo "The Python setup will take a significant amount of time as it"
echo "compiles libraries specifically for your device's CPU."
echo ""
echo "Please ensure your device is connected to a stable Wi-Fi network."
echo "========================================="
read -p "Press [Enter] to begin the installation..."

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
echo "[*] Installing core system dependencies (python, nodejs, git, clang)..."
pkg install -y python git nodejs-lts clang make libjpeg-turbo libcrypt
if [ $? -ne 0 ]; then
    echo "[!] Failed to install core system packages."
    exit 1
fi
echo "[+] Core system dependencies installed successfully."

# --- Install Node.js Dependencies for P2P Bridge ---
# Navigate to the root of the repository to find package.json
cd ..
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
    echo "[!] package.json not found in the root directory. Skipping npm install."
fi
# Navigate back to the android directory
cd android


# --- Delegate to evolve.py for Python Dependencies ---
echo ""
echo "[*] Handing off to evolve.py for platform-aware Python dependency installation..."
echo "[*] This is the longest step and will involve compiling code. Please be patient."
# We run the script with --help. The dependency check runs before arg parsing.
# This avoids starting the full application while still triggering the setup.
python ../evolve.py --help > /dev/null
if [ $? -ne 0 ]; then
    echo "[!] The dependency installer in evolve.py failed."
    echo "[!] Please check the output above for errors."
    exit 1
fi
echo "[+] All Python dependencies installed successfully."


echo ""
echo "========================================="
echo "  SETUP COMPLETE"
echo "========================================="
echo "The environment is ready. You can now run the application using:"
echo ""
echo "  bash run.sh"
echo ""
echo "========================================="