
import pytest
from PIL import Image, ImageStat
import sys
import os

# Ensure we can import from core
sys.path.append(os.getcwd())

from core.text_overlay_utils import analyze_image_region_brightness

def test_analyze_image_region_brightness_performance():
    """
    Verify that analyze_image_region_brightness returns the correct type and runs without error.
    Performance is implicitly checked by the test running quickly.
    """
    # Create a small image for testing correctness
    img = Image.new('RGB', (100, 100), color=(128, 128, 128))

    # Analyze brightness
    brightness = analyze_image_region_brightness(img)

    # Check result
    assert isinstance(brightness, float)
    assert 127.0 <= brightness <= 129.0

def test_analyze_image_region_brightness_empty():
    """Test handling of empty region behavior (simulated by small image)"""
    # Just ensure it doesn't crash on small images
    img = Image.new('RGB', (1, 1), color=(0, 0, 0))
    brightness = analyze_image_region_brightness(img)
    assert isinstance(brightness, float)

def test_analyze_image_region_brightness_white():
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    brightness = analyze_image_region_brightness(img)
    assert brightness > 250

def test_analyze_image_region_brightness_black():
    img = Image.new('RGB', (100, 100), color=(0, 0, 0))
    brightness = analyze_image_region_brightness(img)
    assert brightness < 5
