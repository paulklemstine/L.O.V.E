import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import PostConcept


# US-006: Style to font mapping for concept-driven overlays
STYLE_FONT_MAP = {
    "glitch": ["monospace", "courier", "consolas"],
    "pixel": ["monospace", "terminal"],
    "8-bit": ["monospace", "terminal"],
    "academic": ["times", "georgia", "serif"],
    "baroque": ["times", "georgia", "serif"],
    "renaissance": ["times", "georgia", "serif"],
    "cyber": ["arial", "helvetica", "sans-serif"],
    "neon": ["arial", "helvetica", "sans-serif"],
    "vaporwave": ["arial", "helvetica", "impact"],
    "gothic": ["times", "blackletter"],
    "surfer": ["arial", "impact", "sans-serif"],
}

# --- Subliminal Text Overlay Pools ---
# Font paths for subliminal text - bold, high-impact fonts
SUBLIMINAL_FONT_PATHS = [
    # Windows fonts
    "C:\\Windows\\Fonts\\impact.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\ariblk.ttf",
    "C:\\Windows\\Fonts\\trebucbd.ttf",
    # Linux fonts
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
]

# High-contrast colors for subliminal text
SUBLIMINAL_COLORS = [
    (255, 0, 128),    # Hot Pink / Magenta
    (0, 255, 255),    # Cyan
    (255, 255, 0),    # Yellow
    (0, 255, 0),      # Lime Green
    (255, 165, 0),    # Orange
    (255, 255, 255),  # White
    (0, 0, 0),        # Black
    (255, 0, 0),      # Red
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Electric Blue
]

# Stroke colors that contrast well
SUBLIMINAL_STROKE_COLORS = [
    (0, 0, 0),        # Black
    (255, 255, 255),  # White
    (0, 0, 128),      # Dark Blue
    (128, 0, 0),      # Dark Red
]


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)  # Default white


def get_contrasting_color(bg_colors: list) -> tuple:
    """Get a contrasting color for text visibility against background."""
    if not bg_colors:
        return (255, 255, 255)  # White default
    
    # Calculate average brightness of palette
    total_brightness = 0
    for color in bg_colors[:3]:  # Consider first 3 colors
        if isinstance(color, str):
            rgb = hex_to_rgb(color)
        else:
            rgb = color
        brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
        total_brightness += brightness
    
    avg_brightness = total_brightness / min(len(bg_colors), 3)
    
    # Return white for dark backgrounds, dark for light backgrounds
    if avg_brightness < 128:
        return (255, 255, 255)  # White
    else:
        return (0, 0, 0)  # Black


def analyze_image_region_brightness(image: Image.Image, region: str = "center") -> float:
    """
    Analyze the average brightness of an image region.
    
    Args:
        image: PIL Image to analyze
        region: 'top', 'center', or 'bottom'
        
    Returns:
        Average brightness value (0-255)
    """
    width, height = image.size
    
    # Define region bounds (20% of image height)
    region_height = int(height * 0.2)
    if region == "top":
        box = (0, 0, width, region_height)
    elif region == "bottom":
        box = (0, height - region_height, width, height)
    else:  # center
        center_start = (height - region_height) // 2
        box = (0, center_start, width, center_start + region_height)
    
    # Crop and analyze
    region_img = image.crop(box)
    
    # Convert to grayscale for brightness analysis
    gray = region_img.convert('L')
    
    # Calculate average brightness
    pixels = list(gray.getdata())
    avg_brightness = sum(pixels) / len(pixels) if pixels else 128
    
    return avg_brightness


def get_contrasting_subliminal_colors(image: Image.Image, position: str = "center") -> tuple:
    """
    Select contrasting fill and stroke colors based on image region brightness.
    
    Args:
        image: PIL Image to analyze
        position: 'top', 'center', or 'bottom'
        
    Returns:
        Tuple of (fill_color, stroke_color)
    """
    brightness = analyze_image_region_brightness(image, position)
    
    # For dark backgrounds, use bright colors
    if brightness < 100:
        # Dark background - use bright fill, dark stroke
        bright_colors = [(255, 0, 128), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 165, 0), (255, 255, 255)]
        fill_color = random.choice(bright_colors)
        stroke_color = (0, 0, 0)
    elif brightness > 180:
        # Light background - use dark fill, light stroke
        dark_colors = [(0, 0, 0), (128, 0, 128), (0, 0, 128), (128, 0, 0)]
        fill_color = random.choice(dark_colors)
        stroke_color = (255, 255, 255)
    else:
        # Medium brightness - use high contrast pairs
        contrast_pairs = [
            ((255, 255, 255), (0, 0, 0)),
            ((255, 255, 0), (0, 0, 0)),
            ((0, 255, 255), (0, 0, 0)),
            ((255, 0, 128), (255, 255, 255)),
        ]
        fill_color, stroke_color = random.choice(contrast_pairs)
    
    return fill_color, stroke_color


def overlay_text_from_concept(
    image: Image.Image,
    concept: 'PostConcept',
    position: str = "bottom"
) -> Image.Image:
    """
    US-006: Overlays text onto an image using PostConcept for styling.
    
    Uses subliminal_intent or key_message for text content.
    Selects font style based on visual_style.
    Ensures contrast visibility against color_palette.
    
    Args:
        image: Source PIL Image
        concept: PostConcept with styling information
        position: Position of text (header, footer, center, bottom)
        
    Returns:
        Modified PIL Image with concept-driven overlay
    """
    # Determine text content - prefer subliminal_intent, fallback to key_message
    text = concept.subliminal_intent
    if not text or len(text) > 50:
        # Use first part of key_message if subliminal is too long
        text = concept.key_message[:40] if concept.key_message else "✨"
    
    if not text:
        return image
    
    logging.info(f"Applying concept overlay: text='{text}', style='{concept.visual_style[:30]}...'")
    
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # --- Font Style Selection Based on visual_style ---
    font_size = int(width / 15)
    style_lower = concept.visual_style.lower()
    
    # Determine preferred font type based on visual style
    preferred_fonts = []
    for style_key, fonts in STYLE_FONT_MAP.items():
        if style_key in style_lower:
            preferred_fonts = fonts
            break
    
    # Font path mapping
    # Robust asset path resolution relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_fonts = os.path.join(base_dir, "assets", "fonts")
    
    font_paths = [
        os.path.join(assets_fonts, "arialbd.ttf"),
        "assets/fonts/arialbd.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\impact.ttf",
        "C:\\Windows\\Fonts\\seguiemj.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "arial.ttf"
    ]
    
    # Try monospace fonts first if style requires it
    if any(pref in ["monospace", "courier", "terminal"] for pref in preferred_fonts):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "C:\\Windows\\Fonts\\consola.ttf",
        ] + font_paths
    
    font = None
    for path in font_paths:
        try:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        except Exception:
            continue
    
    if not font:
        logging.warning("Could not load fonts, using default.")
        font = ImageFont.load_default()
    
    # --- Position Calculation ---
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Shrink if too wide
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
    
    x = (width - text_width) // 2
    y = int(height * 0.8) - (text_height // 2)
    
    padding = int(height * 0.05)
    if position == "header":
        y = padding
    elif position == "center":
        y = (height - text_height) // 2
    
    # --- Color Selection from Concept ---
    # Get contrasting text color based on palette
    fill_color = get_contrasting_color(concept.color_palette)
    
    # Use first palette color for stroke if available
    stroke_color = (0, 0, 0)  # Default black stroke
    if concept.color_palette:
        first_color = concept.color_palette[0]
        if isinstance(first_color, str):
            stroke_color = hex_to_rgb(first_color)
    
    # Invert if contrasting gives same result
    if fill_color == stroke_color:
        stroke_color = (0, 0, 0) if fill_color == (255, 255, 255) else (255, 255, 255)
    
    # --- Render with stroke for visibility ---
    stroke_width = 2
    draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
    
    logging.info(f"Concept overlay applied: fill={fill_color}, stroke={stroke_color}")
    return image


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
    
    # Robust asset path resolution relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_fonts = os.path.join(base_dir, "assets", "fonts")
    local_font_path = os.path.join(assets_fonts, "arialbd.ttf")
    
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
    
    if style == "subliminal":
        # SUBLIMINAL STYLE: Smart contrast + random font from pool
        # Analyze image region to get contrasting colors
        fill_color, stroke_color = get_contrasting_subliminal_colors(image, position)
        stroke_width = 3  # Thicker stroke for subliminal visibility
        
        # Try to use a random font from the subliminal pool
        subliminal_font = None
        random.shuffle(SUBLIMINAL_FONT_PATHS)  # Randomize font selection
        for path in SUBLIMINAL_FONT_PATHS:
            try:
                if os.path.exists(path):
                    subliminal_font = ImageFont.truetype(path, font_size)
                    logging.info(f"Subliminal using font: {path}")
                    break
            except Exception:
                continue
        
        if subliminal_font:
            font = subliminal_font
            # Recalculate text dimensions with new font
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Shrink font if text is too wide for the image
            current_font_size = font_size
            while text_width > width * 0.9 and current_font_size > 10:
                current_font_size -= 2
                try:
                    if hasattr(font, 'path'):
                        font = ImageFont.truetype(font.path, current_font_size)
                    else:
                        break
                except:
                    break
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            
            x = (width - text_width) // 2
            
            # Recalculate Y position based on position argument
            if position == "top":
                y = padding
            elif position == "center":
                y = (height - text_height) // 2
            else:  # bottom
                y = int(height * 0.8) - (text_height // 2)
        
        logging.info(f"Subliminal overlay: fill={fill_color}, stroke={stroke_color}, pos={position}")
        draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
        
    elif style == "neon":
        # Stroke
        stroke_width = 2
        stroke_color = (0, 255, 255)  # Cyan stroke for neon
        fill_color = (255, 255, 255)
        
        draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color)
        
    elif style == "meme":
        draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        
    else:
        # Default
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
    return image
