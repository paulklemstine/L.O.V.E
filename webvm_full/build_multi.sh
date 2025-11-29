#!/bin/bash
set -e

echo "=== Multi-Volume WebVM Build Script ==="
echo "This will create 3 separate ext2 images:"
echo "  1. base.ext2 (OS + Python + tools)"
echo "  2. packages.ext2 (Python packages)"
echo "  3. app.ext2 (L.O.V.E code)"
echo ""

# Get the repository root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repository root: $REPO_ROOT"

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -f base.tar base.ext2
rm -f packages.tar packages.ext2
rm -f app.tar app.ext2
rm -rf src/

# === BUILD BASE SYSTEM ===
echo ""
echo "=== Building Base System Image ==="
docker build --platform linux/386 -f Dockerfile.base -t love-webvm-base .

echo "Exporting base filesystem..."
id=$(docker create --platform linux/386 love-webvm-base)
docker export $id > base.tar
docker rm $id

echo "Creating base.ext2 (2GB)..."
dd if=/dev/zero of=base.ext2 bs=1M count=2048
sudo mkfs.ext2 -F base.ext2
mkdir -p /tmp/base_mount
sudo mount -o loop base.ext2 /tmp/base_mount
sudo tar -xf base.tar -C /tmp/base_mount
sudo umount /tmp/base_mount
rmdir /tmp/base_mount
rm base.tar

BASE_SIZE=$(du -h base.ext2 | cut -f1)
echo "✓ Base system created: $BASE_SIZE"

# === BUILD PACKAGES VOLUME ===
echo ""
echo "=== Building Packages Volume ==="
docker build --platform linux/386 -f Dockerfile.packages -t love-webvm-packages .

echo "Exporting packages filesystem..."
id=$(docker create --platform linux/386 love-webvm-packages)
docker export $id > packages.tar
docker rm $id

echo "Creating packages.ext2 (2GB)..."
dd if=/dev/zero of=packages.ext2 bs=1M count=2048
sudo mkfs.ext2 -F packages.ext2
mkdir -p /tmp/packages_mount
sudo mount -o loop packages.ext2 /tmp/packages_mount
sudo tar -xf packages.tar -C /tmp/packages_mount
sudo umount /tmp/packages_mount
rmdir /tmp/packages_mount
rm packages.tar

PACKAGES_SIZE=$(du -h packages.ext2 | cut -f1)
echo "✓ Packages volume created: $PACKAGES_SIZE"

# === BUILD APP VOLUME ===
echo ""
echo "=== Building App Volume ==="

# Prepare source directory for app
echo "Preparing L.O.V.E source..."
mkdir -p src
rsync -av --exclude 'webvm_full' --exclude '.git' --exclude '*.ext2' --exclude '*.tar' "$REPO_ROOT/" src/

docker build --platform linux/386 -f Dockerfile.app -t love-webvm-app src/

echo "Exporting app filesystem..."
id=$(docker create --platform linux/386 love-webvm-app)
docker export $id > app.tar
docker rm $id

echo "Creating app.ext2 (512MB)..."
dd if=/dev/zero of=app.ext2 bs=1M count=512
sudo mkfs.ext2 -F app.ext2
mkdir -p /tmp/app_mount
sudo mount -o loop app.ext2 /tmp/app_mount
sudo tar -xf app.tar -C /tmp/app_mount
sudo umount /tmp/app_mount
rmdir /tmp/app_mount
rm app.tar

APP_SIZE=$(du -h app.ext2 | cut -f1)
echo "✓ App volume created: $APP_SIZE"

# === SUMMARY ===
echo ""
echo "=== Build Complete ==="
echo "Created volumes:"
echo "  base.ext2:     $BASE_SIZE"
echo "  packages.ext2: $PACKAGES_SIZE"
echo "  app.ext2:      $APP_SIZE"
echo ""
echo "Starting local server on port 8000..."
echo "Open http://localhost:8000 in your browser."
python3 server.py 8000
