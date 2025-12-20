import sys
import os
from PIL import ImageFont, ImageDraw, Image

# Add current directory to path
sys.path.append(os.getcwd())

from core.text_overlay_utils import overlay_text_on_image

def test_font_loading():
    print("Testing WSL Font Loading...")
    
    # Create a dummy image
    img = Image.new("RGB", (500, 500), (0, 0, 0))
    
    # Try overlaying text
    try:
        # We need to access the internal font loading logic or just check if the result "looks" correct 
        # (difficult programmatically for "looks"), but we can patch ImageFont.truetype to verify it's called.
        
        from PIL import ImageFont
        original_truetype = ImageFont.truetype
        
        loaded_path = []
        
        def side_effect(font, size, index=0, encoding="", layout_engine=None):
            loaded_path.append(font)
            return original_truetype(font, size, index, encoding, layout_engine)
            
        ImageFont.truetype = side_effect
        
        overlay_text_on_image(img, "TEST FONT", position="center")
        
        print(f"Fonts attempted: {loaded_path}")
        
        if any("/mnt/c/Windows/Fonts" in p or "/usr/share/fonts" in p for p in loaded_path):
             print(f"SUCCESS: Successfully attempted to load a system font: {loaded_path}")
        else:
             print("WARNING: Only attempted local/default fonts?")

    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_font_loading()
