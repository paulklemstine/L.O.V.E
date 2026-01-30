
import time
from PIL import Image, ImageStat
import random

def old_analyze_image_region_brightness(image, region="center"):
    width, height = image.size

    # Define region bounds (20% of image height)
    region_height = int(height * 0.2)
    if region == "top":
        box = (0, 0, width, region_height)
    elif region == "bottom":
        box = (0, height - region_height, width, height)
    else:  # center
        center_start = (height - region_height) // 2
        box = (0, center_start, width, center_start + region_height)

    # Crop and analyze
    region_img = image.crop(box)

    # Convert to grayscale for brightness analysis
    gray = region_img.convert('L')

    # Calculate average brightness
    pixels = list(gray.getdata())
    avg_brightness = sum(pixels) / len(pixels) if pixels else 128

    return avg_brightness

def new_analyze_image_region_brightness(image, region="center"):
    width, height = image.size

    # Define region bounds (20% of image height)
    region_height = int(height * 0.2)
    if region == "top":
        box = (0, 0, width, region_height)
    elif region == "bottom":
        box = (0, height - region_height, width, height)
    else:  # center
        center_start = (height - region_height) // 2
        box = (0, center_start, width, center_start + region_height)

    # Crop and analyze
    region_img = image.crop(box)

    # Convert to grayscale for brightness analysis
    gray = region_img.convert('L')

    # Calculate average brightness using ImageStat
    stat = ImageStat.Stat(gray)
    avg_brightness = stat.mean[0] if stat.count[0] > 0 else 128.0

    return avg_brightness

def run_benchmark():
    # Create a large image
    print("Creating test image...")
    img = Image.new('RGB', (4000, 4000), color=(128, 128, 128))

    # Warm up
    print("Warming up...")
    old_analyze_image_region_brightness(img)
    new_analyze_image_region_brightness(img)

    # Benchmark Old
    start_time = time.time()
    for _ in range(10):
        old_analyze_image_region_brightness(img)
    old_time = time.time() - start_time
    print(f"Old method (10 runs): {old_time:.4f}s")

    # Benchmark New
    start_time = time.time()
    for _ in range(10):
        new_analyze_image_region_brightness(img)
    new_time = time.time() - start_time
    print(f"New method (10 runs): {new_time:.4f}s")

    # Improvement
    if new_time > 0:
        print(f"Speedup: {old_time / new_time:.2f}x")

    # Verification
    v1 = old_analyze_image_region_brightness(img)
    v2 = new_analyze_image_region_brightness(img)
    print(f"Values match: {abs(v1 - v2) < 0.1} ({v1} vs {v2})")

if __name__ == "__main__":
    run_benchmark()
