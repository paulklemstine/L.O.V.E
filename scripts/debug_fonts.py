
import os
from PIL import Image, ImageDraw, ImageFont

def test_font_loading():
    print("Checking for fonts...")
    
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/example/font.ttf" # Dummy
    ]
    
    found_font = None
    for path in font_paths:
        if os.path.exists(path):
            print(f"FOUND: {path}")
            found_font = path
            break
        else:
            print(f"MISSING: {path}")
            
    if found_font:
        try:
            font = ImageFont.truetype(found_font, 40)
            print("Successfully loaded TrueType font.")
        except Exception as e:
            print(f"Failed to load TrueType font: {e}")
            font = ImageFont.load_default()
    else:
        print("No TrueType fonts found. Falling back to default.")
        font = ImageFont.load_default()
        
    # Create test image
    img = Image.new("RGB", (500, 200), "black")
    draw = ImageDraw.Draw(img)
    
    # Draw text
    try:
        draw.text((10, 50), "TEST RENDER", font=font, fill="pink")
        output_path = "generated_images/debug_overlay_test.png"
        img.save(output_path)
        print(f"Saved test image to {output_path}")
    except Exception as e:
        print(f"Drawing failed: {e}")

if __name__ == "__main__":
    test_font_loading()
