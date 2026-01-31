import time
from PIL import Image, ImageStat, ImageDraw
import random
import sys
import os

# Add repo root to path
sys.path.append(os.getcwd())

from core.text_overlay_utils import analyze_image_region_brightness

# Legacy implementation for comparison
def analyze_image_region_brightness_legacy(image: Image.Image, region: str = "center") -> float:
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

    # Calculate average brightness using OLD METHOD
    pixels = list(gray.getdata())
    avg_brightness = sum(pixels) / len(pixels) if pixels else 128

    return avg_brightness

def benchmark():
    # Create a large random image
    width, height = 2048, 2048
    print(f"Generating {width}x{height} image...")
    image = Image.new('RGB', (width, height), color='red')

    # Add some noise so it's not uniform
    draw = ImageDraw.Draw(image)
    for _ in range(1000):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    print(f"Benchmarking...")

    # Correctness check
    print("Verifying correctness...")
    val_optimized = analyze_image_region_brightness(image, "center") # This is now the optimized one from the file
    val_legacy = analyze_image_region_brightness_legacy(image, "center")
    print(f"Legacy Value: {val_legacy}")
    print(f"Optimized Value: {val_optimized}")

    if abs(val_legacy - val_optimized) > 0.1:
        print("ERROR: Values differ too much!")
        sys.exit(1)
    print("Correctness verified.")

    iterations = 20

    # Measure legacy method
    start_time = time.time()
    for _ in range(iterations):
        analyze_image_region_brightness_legacy(image, "center")
    legacy_duration = time.time() - start_time
    print(f"Legacy method ({iterations} runs): {legacy_duration:.4f}s")

    # Measure optimized method (from file)
    start_time = time.time()
    for _ in range(iterations):
        analyze_image_region_brightness(image, "center")
    optimized_duration = time.time() - start_time
    print(f"Optimized method (from file) ({iterations} runs): {optimized_duration:.4f}s")

    if legacy_duration > 0:
        improvement = (legacy_duration - optimized_duration) / legacy_duration * 100
        print(f"Improvement: {improvement:.2f}%")
        print(f"Speedup: {legacy_duration/optimized_duration:.2f}x")

if __name__ == "__main__":
    benchmark()
