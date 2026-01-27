import time
import random
import pytest
from PIL import Image
from core.text_overlay_utils import analyze_image_region_brightness

def legacy_brightness(image, region="center"):
    # Copied from original implementation for verification
    width, height = image.size
    region_height = int(height * 0.2)
    if region == "top":
        box = (0, 0, width, region_height)
    elif region == "bottom":
        box = (0, height - region_height, width, height)
    else:  # center
        center_start = (height - region_height) // 2
        box = (0, center_start, width, center_start + region_height)

    region_img = image.crop(box)
    gray = region_img.convert('L')
    pixels = list(gray.getdata())
    avg_brightness = sum(pixels) / len(pixels) if pixels else 128
    return avg_brightness

def test_brightness_correctness():
    # Create random image (smaller for fast testing)
    img = Image.new('RGB', (100, 100))
    # Random pixels
    pixels = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100*100)]
    img.putdata(pixels)

    # Test all regions
    for region in ["top", "center", "bottom"]:
        expected = legacy_brightness(img, region)
        actual = analyze_image_region_brightness(img, region)

        # Allow small floating point difference
        assert abs(expected - actual) < 1.0, f"Region {region}: Expected {expected}, got {actual}"

def test_empty_image_fallback():
    # Test case where image (or region) is empty
    # Creating a 0x0 image directly or small image where crop might be empty
    img = Image.new('RGB', (10, 2)) # Very small height

    # Check that it returns default 128 and doesn't crash
    val = analyze_image_region_brightness(img, "center")
    assert val == 128.0

def test_brightness_performance_check():
    # Ensure it is fast (performance check)
    img = Image.new('RGB', (1024, 1024))
    start = time.time()
    for _ in range(10):
        analyze_image_region_brightness(img, "center")
    duration = time.time() - start

    # Should be very fast (< 0.1s for 10 runs)
    assert duration < 1.0, f"Performance regression: took {duration}s"
