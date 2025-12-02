#!/bin/bash
set -e

echo "=== Multi-Volume WebVM Build Script ==="
echo "This will create 3 separate ext2 images:"
echo "  1. base.ext2 (2GB - minimal OS + Python)"
echo "  2. packages.ext2 (2GB - Python packages)"
echo "  3. app.ext2 (512MB - L.O.V.E code)"
echo ""

# Get the repository root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repository root: $REPO_ROOT"

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -f base.tar base.ext2
rm -f packages.tar packages.ext2
rm -f app.tar app.ext2
rm -f requirements.txt
rm -rf src/

# === BUILD BASE SYSTEM ===
echo ""
echo "=== Building Base System Image ==="
docker build --platform linux/386 -f Dockerfile.base -t love-webvm-base .

echo "Creating base.ext2 (2GB)..."
dd if=/dev/zero of=base.ext2 bs=1M count=2048
sudo mkfs.ext2 -F base.ext2
mkdir -p /tmp/base_mount
sudo mount -o loop base.ext2 /tmp/base_mount

echo "Exporting and extracting base filesystem..."
id=$(docker create --platform linux/386 love-webvm-base)
docker export $id | sudo tar -x -C /tmp/base_mount
docker rm $id

sudo umount /tmp/base_mount
sudo umount -f /tmp/base_mount 2>/dev/null || true  # Force unmount if still busy
rmdir /tmp/base_mount

BASE_SIZE=$(du -h base.ext2 | cut -f1)
echo "✓ Base system created: $BASE_SIZE"

# === BUILD PACKAGES VOLUME ===
echo ""
echo "=== Building Packages Volume ==="

# Copy requirements.txt to current directory for Docker build context
cp "$REPO_ROOT/requirements.txt" .
# Remove triton (not supported on 32-bit)
sed -i '/triton/d' requirements.txt
# Remove torch and related heavy ML packages (installed in base or skipped if unsupported)
sed -i '/torch/d' requirements.txt
sed -i '/accelerate/d' requirements.txt
sed -i '/transformers/d' requirements.txt
sed -i '/llmlingua/d' requirements.txt
sed -i '/faiss-cpu/d' requirements.txt
sed -i '/pyarrow/d' requirements.txt
sed -i '/hf-xet/d' requirements.txt
sed -i '/pywin32/d' requirements.txt
sed -i '/pyreadline3/d' requirements.txt

docker build --platform linux/386 -f Dockerfile.packages -t love-webvm-packages .

echo "Creating packages.ext2 (2GB)..."
dd if=/dev/zero of=packages.ext2 bs=1M count=2048
sudo mkfs.ext2 -F packages.ext2
mkdir -p /tmp/packages_mount
sudo mount -o loop packages.ext2 /tmp/packages_mount

echo "Exporting and extracting packages filesystem..."
id=$(docker create --platform linux/386 love-webvm-packages true)
docker export $id | sudo tar -x -C /tmp/packages_mount
docker rm $id

sudo umount /tmp/packages_mount
sudo umount -f /tmp/packages_mount 2>/dev/null || true  # Force unmount if still busy
rmdir /tmp/packages_mount

PACKAGES_SIZE=$(du -h packages.ext2 | cut -f1)
echo "✓ Packages volume created: $PACKAGES_SIZE"

# === BUILD APP VOLUME ===
echo ""
echo "=== Building App Volume ==="

# Prepare source directory for app
echo "Preparing L.O.V.E source..."
mkdir -p src
rsync -av \
    --exclude 'webvm_full' \
    --exclude '.git' \
    --exclude '*.ext2' \
    --exclude '*.tar' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.ipfs_repo' \
    --exclude 'generated_images' \
    --exclude '*.log' \
    --exclude 'vllm_server.log' \
    --exclude 'love.log' \
    --exclude 'ipfs.log' \
    --exclude '.pytest_cache' \
    --exclude 'build' \
    --exclude 'dist' \
    --exclude '*.egg-info' \
    "$REPO_ROOT/" src/

# Copy the run_love.sh script into the app source
echo "Adding run_love.sh startup script..."
cp run_love.sh src/run_love.sh
chmod +x src/run_love.sh

docker build --platform linux/386 -f Dockerfile.app -t love-webvm-app src/

echo "Creating app.ext2 (512MB)..."
dd if=/dev/zero of=app.ext2 bs=1M count=512
sudo mkfs.ext2 -F app.ext2
mkdir -p /tmp/app_mount
sudo mount -o loop app.ext2 /tmp/app_mount

echo "Exporting and extracting app filesystem..."
id=$(docker create --platform linux/386 love-webvm-app true)
docker export $id | sudo tar -x -C /tmp/app_mount
docker rm $id

sudo umount /tmp/app_mount
sudo umount -f /tmp/app_mount 2>/dev/null || true  # Force unmount if still busy
rmdir /tmp/app_mount

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
