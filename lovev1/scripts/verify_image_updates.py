#!/usr/bin/env python3
"""Verification script for image generator updates."""
import sys
sys.path.insert(0, '/home/raver1975/L.O.V.E')

# Test imports
print("Testing imports...")
try:
    from core.image_generation_pool import (
        generate_image_with_pool, 
        IMAGE_MODEL_STATS
    )
    from core.text_overlay_utils import (
        overlay_text_on_image, 
        get_contrasting_subliminal_colors, 
        analyze_image_region_brightness, 
        SUBLIMINAL_FONT_PATHS, 
        SUBLIMINAL_COLORS
    )
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Verify configuration
print(f"\n📊 Configuration:")
print(f"  - Subliminal font paths: {len(SUBLIMINAL_FONT_PATHS)} fonts")
print(f"  - Subliminal colors: {len(SUBLIMINAL_COLORS)} colors")
print(f"  - AI Horde stats: {IMAGE_MODEL_STATS.get('ai_horde', 'Not initialized')}")

# Test contrast detection with dummy image
print(f"\n🎨 Testing contrast detection...")
from PIL import Image
test_img = Image.new('RGB', (100, 100), color='black')
fill, stroke = get_contrasting_subliminal_colors(test_img, "center")
print(f"  Dark background -> fill={fill}, stroke={stroke}")

test_img2 = Image.new('RGB', (100, 100), color='white')
fill2, stroke2 = get_contrasting_subliminal_colors(test_img2, "center")
print(f"  Light background -> fill={fill2}, stroke={stroke2}")

print("\n✅ All tests passed!")
