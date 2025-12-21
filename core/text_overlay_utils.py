import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random

def overlay_text_on_image(image: Image.Image, text: str, position: str = "bottom", style: str = "neon") -> Image.Image:
    """
    Overlays text onto an image using PIL.
    
    Args:
        image: Source PIL Image
        text: Text to overlay
        position: Position of text (header, footer, center, center_left, etc.)
        style: Rendering style (neon, meme)
        
    Returns:
        Modified PIL Image
    """
    if not text:
        return image
        
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # --- Font Selection ---
    # Smart scaling: width / 15 as per Story 2.3
    font_size = int(width / 15)
    
    local_font_path = os.path.join(os.getcwd(), "assets", "fonts", "arialbd.ttf")
    # Extended list of likely system fonts
    font_paths = [
        local_font_path,
        "assets/fonts/arialbd.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\impact.ttf",
        "C:\\Windows\\Fonts\\seguiemj.ttf", # Segoe UI Emoji usually exists
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arial.ttf"
    ]
    
    font = None
    for path in font_paths:
        try:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        except Exception:
            continue
            
    if not font:
        # Fallback to default if absolutely nothing is found
        # Note: Default font doesn't support size scaling well usually
        logging.warning("Could not load preferred fonts, falling back to default.")
        font = ImageFont.load_default()

    # --- Position Calculation ---
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Safety check: if text is still too wide, shrink it
    while text_width > width * 0.9 and font_size > 10:
        font_size -= 2
        try:
            if hasattr(font, 'path'):
                font = ImageFont.truetype(font.path, font_size)
            else:
                 break 
        except:
             break
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

    # Calculate Coordinates
    # Center Horizontally
    x = (width - text_width) // 2
    
    # Story 2.3 Requirement: Lower third (e.g., 20% from bottom)
    # y = height - (height * 0.20) - (text_height / 2)
    # Let's place the baseline at 80% down implies the text sits around there.
    y = int(height * 0.8) - (text_height // 2)
    
    # Overrides
    padding = int(height * 0.05)
    if position == "header":
        y = padding
    elif position == "center":
        y = (height - text_height) // 2
    # "bottom" or "lower_third" falls through to the calculated Y above

    # --- Rendering Styles ---
    # Story 2.3 Requirement: Black stroke (2px) + hot pink/main color fill
    
    if style == "neon" or style == "subliminal":
        # Stroke
        stroke_width = 2
        stroke_color = (0, 0, 0)
        
        # Fill - Hot Pink for high visibility/Ero-Kakkoii
        fill_color = (255, 105, 180) # Hot Pink
        if style == "neon":
             # Keeps cyan glow for "neon" style if requested
             fill_color = (255, 255, 255)
             stroke_color = (0, 255, 255) # Cyan stroke for neon
        
        draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
        
    elif style == "meme":
        draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        
    else:
        # Default
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
    return image
