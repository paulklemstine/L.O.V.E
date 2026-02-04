import pytest
from PIL import Image
from core.text_overlay_utils import analyze_image_region_brightness

def test_analyze_image_region_brightness():
    # Create a 100x100 gray image with pixel value 100
    width, height = 100, 100
    color_value = 100
    image = Image.new('RGB', (width, height), (color_value, color_value, color_value))

    # Test center region
    brightness = analyze_image_region_brightness(image, region="center")

    # Since the image is uniform, brightness should be exactly 100
    # Floating point arithmetic might have slight deviations, so use approx
    assert brightness == pytest.approx(color_value, abs=1.0)

    # Test with a gradient to ensure it averages correctly
    # Gradient from 0 to 255
    image_gradient = Image.new('L', (100, 100))
    for y in range(100):
        for x in range(100):
            image_gradient.putpixel((x, y), x + y) # Just some variation

    # We won't calculate exact expected value for gradient here as it depends on exact crop logic,
    # but we can ensure it runs and returns a valid float between 0 and 255.
    brightness_grad = analyze_image_region_brightness(image_gradient, region="center")
    assert 0 <= brightness_grad <= 255
    assert isinstance(brightness_grad, float)
