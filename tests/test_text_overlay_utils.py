
import unittest
from PIL import Image
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from core.text_overlay_utils import analyze_image_region_brightness

class TestTextOverlayUtils(unittest.TestCase):
    def test_analyze_image_region_brightness(self):
        # 1. Solid Black Image
        black_img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        brightness = analyze_image_region_brightness(black_img, "center")
        self.assertEqual(brightness, 0.0, "Black image should have 0 brightness")

        # 2. Solid White Image
        white_img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        brightness = analyze_image_region_brightness(white_img, "center")
        self.assertEqual(brightness, 255.0, "White image should have 255 brightness")

        # 3. Solid Gray Image
        gray_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        brightness = analyze_image_region_brightness(gray_img, "center")
        self.assertAlmostEqual(brightness, 128.0, delta=1.0, msg="Gray image should have ~128 brightness")

        # 4. Mixed Image (half black, half white) -> Avg 127.5
        # Since we crop 20% height, we should ensure the crop region has the mix.
        # Let's make it vertical stripes so any horizontal crop gets the mix.
        mixed_img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        # Draw white on right half
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mixed_img)
        draw.rectangle([(50, 0), (100, 100)], fill=(255, 255, 255))

        brightness = analyze_image_region_brightness(mixed_img, "center")
        self.assertAlmostEqual(brightness, 127.5, delta=1.0, msg="Half/Half image should have ~127.5 brightness")

if __name__ == '__main__':
    unittest.main()
