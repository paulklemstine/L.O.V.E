#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t love-cheerpx .

echo "Exporting filesystem..."
id=$(docker create love-cheerpx)
docker export $id > love.tar
docker rm $id

echo "Creating ext2 image (requires sudo for mount)..."
# Create a 512MB empty file
dd if=/dev/zero of=love.ext2 bs=1M count=512

# Format as ext2
mkfs.ext2 love.ext2

echo "Populating ext2 image..."
mkdir -p mnt
sudo mount -o loop love.ext2 mnt
sudo tar -xf love.tar -C mnt
sudo umount mnt
rmdir mnt
rm love.tar

echo "Build complete: love.ext2 created."
echo "Starting local server on port 8000..."
echo "Open http://localhost:8000 in your browser."
python3 -m http.server 8000
