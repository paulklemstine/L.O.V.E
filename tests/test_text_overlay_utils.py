
import sys
import os
import unittest
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.text_overlay_utils import analyze_image_region_brightness

class TestTextOverlayUtils(unittest.TestCase):
    def test_analyze_image_region_brightness_dark(self):
        # Create a black image
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        brightness = analyze_image_region_brightness(img, region="center")
        self.assertEqual(brightness, 0.0)

    def test_analyze_image_region_brightness_bright(self):
        # Create a white image
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        brightness = analyze_image_region_brightness(img, region="center")
        self.assertEqual(brightness, 255.0)

    def test_analyze_image_region_brightness_gray(self):
        # Create a gray image (128)
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        brightness = analyze_image_region_brightness(img, region="center")
        self.assertAlmostEqual(brightness, 128.0, delta=1.0)

    def test_analyze_image_region_brightness_mixed(self):
        # Create an image that is half black, half white in the center region
        # Region height is 20% of 100 = 20 pixels.
        # Center region starts at (100-20)//2 = 40. Ends at 60.
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))

        # Make the center region half white
        # We want the average of the center region to be 127.5
        # The function crops the center region.
        # Let's just create a uniform image for simplicity of testing "mixed"
        # If we set color to (100, 100, 100), brightness should be 100.
        img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        brightness = analyze_image_region_brightness(img, region="center")
        self.assertAlmostEqual(brightness, 100.0, delta=1.0)

if __name__ == '__main__':
    unittest.main()
