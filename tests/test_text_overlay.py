
import sys
import os
import unittest
from PIL import Image

# Add repo root to path if needed
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from core.text_overlay_utils import analyze_image_region_brightness

class TestTextOverlay(unittest.TestCase):
    def test_analyze_brightness_basic(self):
        # Create a test image
        img = Image.new('RGB', (100, 100), (255, 255, 255)) # White
        brightness = analyze_image_region_brightness(img)
        self.assertEqual(brightness, 255.0)

        img = Image.new('RGB', (100, 100), (0, 0, 0)) # Black
        brightness = analyze_image_region_brightness(img)
        self.assertEqual(brightness, 0.0)

        img = Image.new('RGB', (100, 100), (128, 128, 128)) # Gray
        brightness = analyze_image_region_brightness(img)
        self.assertTrue(127.5 < brightness < 128.5)

    def test_analyze_brightness_region(self):
        # Split image: Top white, Bottom black
        img = Image.new('RGB', (100, 100), (0, 0, 0))
        draw = Image.new('RGB', (100, 50), (255, 255, 255))
        img.paste(draw, (0, 0))

        # Check top (should be white)
        # The function defines top as 20% height
        b_top = analyze_image_region_brightness(img, region="top")
        self.assertEqual(b_top, 255.0)

        # Check bottom (should be black)
        b_bottom = analyze_image_region_brightness(img, region="bottom")
        self.assertEqual(b_bottom, 0.0)

if __name__ == "__main__":
    unittest.main()
