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
    # Try to load a reasonable font
    # PRIORITIZE: Local assets for portability
    local_font_path = os.path.join(os.getcwd(), "assets", "fonts", "arialbd.ttf")
    
    font_paths = [
        local_font_path,
        "assets/fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/mnt/c/Windows/Fonts/arialbd.ttf",
        "/mnt/c/Windows/Fonts/impact.ttf",
        "arial.ttf",
        "Arial.ttf"
    ]
    
    font = None
    font_size = int(height * 0.10) # Start with 10% of image height
    
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
            
    if not font:
        logging.warning("Could not load preferred fonts, falling back to default.")
        font = ImageFont.load_default()
    
    # --- Position Calculation ---
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Scale down if too wide
    while text_width > width * 0.9 and font_size > 10:
        font_size -= 2
        try:
            if isinstance(font, ImageFont.FreeTypeFont):
                font = ImageFont.truetype(font.path, font_size)
            else:
                 # Default font can't resize effectively this way easily without reloading
                 break 
        except:
             break
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

    # Calculate Coordinates
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    padding = int(height * 0.05)
    
    if position == "header" or position == "top":
        y = padding
    elif position == "footer" or position == "bottom":
        y = height - text_height - padding
    elif position == "center":
        pass # Already calculated
    elif position == "center_left":
        x = padding
    elif position == "center_right":
        x = width - text_width - padding
    elif position == "top_left":
        x = padding
        y = padding
    elif position == "bottom_right":
        x = width - text_width - padding
        y = height - text_height - padding
    
    # --- Rendering Styles ---
    
    if style == "neon":
        # Glow effect
        glow_color = (0, 255, 255) # Cyan glow
        text_color = (255, 255, 255)
        
        # Draw multiple offsets for glow
        for offset in range(1, 4):
            draw.text((x-offset, y), text, font=font, fill=glow_color)
            draw.text((x+offset, y), text, font=font, fill=glow_color)
            draw.text((x, y-offset), text, font=font, fill=glow_color)
            draw.text((x, y+offset), text, font=font, fill=glow_color)
            
        draw.text((x, y), text, font=font, fill=text_color)
        
    elif style == "meme":
        # White text, black broad outline
        stroke_width = 3
        stroke_fill = (0, 0, 0)
        text_fill = (255, 255, 255)
        
        draw.text((x, y), text, font=font, fill=text_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        
    else:
        # Default simple
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
    return image
