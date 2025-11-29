#!/bin/bash
set -e

# Define paths
REPO_ROOT=$(pwd)/..
BUILD_DIR=$(pwd)

echo "Preparing build context..."
# Create a temporary 'src' directory to hold the repo files for the build context
rm -rf src
mkdir -p src
# Copy everything from the repo root to src, excluding webvm_full itself to avoid recursion
rsync -av --exclude 'webvm_full' --exclude '.git' --exclude 'love.ext2' "$REPO_ROOT/" src/

echo "Building Docker image (this will take a long time)..."
docker build --platform linux/386 -t love-webvm-full .

echo "Exporting filesystem..."
id=$(docker create --platform linux/386 love-webvm-full)
docker export $id > love.tar
docker rm $id

echo "Creating ext2 image (requires sudo)..."
# Create a 4GB empty file (Torch is heavy)
dd if=/dev/zero of=love.ext2 bs=1M count=4096

# Format as ext2
mkfs.ext2 love.ext2

echo "Populating ext2 image..."
mkdir -p mnt
sudo mount -o loop love.ext2 mnt
sudo tar -xf love.tar -C mnt
sudo umount mnt
rmdir mnt
rm love.tar
rm -rf src

echo "Build complete: love.ext2 created."
echo "Starting local server on port 8000..."
echo "Open http://localhost:8000 in your browser."
python3 server.py 8000
