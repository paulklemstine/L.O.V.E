"""
Intelligent Watermark System with Energy-Based Placement

This module provides sophisticated watermarking capabilities:
- Energy-based optimal placement using Sobel gradients
- Rotating logo pool for variety
- Dynamic text transformations (rotate, scale, skew) based on image content
- Logo + "@e-v-l-o-v-e.bsky.social" text composite watermarks
"""

import os
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from scipy import ndimage
from typing import Tuple, Optional, List
import core.logging

# --- Configuration ---
WATERMARK_OPACITY = 0.25  # Default opacity (0-1)
WATERMARK_MAX_SIZE_RATIO = 0.15  # Max watermark size relative to image
WATERMARK_MIN_SIZE_RATIO = 0.08  # Min watermark size relative to image
WATERMARK_TEXT = "@e-v-l-o-v-e.bsky.social"
WATERMARK_MARGIN = 20  # Pixels from edge

# Logo rotation state file
LOGO_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "watermark_state.json")

# --- Logo Pool ---
def get_logo_paths() -> List[str]:
    """Get all available logo paths from the assets/logos directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logos_dir = os.path.join(base_dir, "assets", "logos")
    
    # Also check root directory for legacy logos
    root_logo = os.path.join(base_dir, "love_logo.png")
    assets_logo = os.path.join(base_dir, "assets")
    
    logo_paths = []
    
    # Add from logos directory
    if os.path.exists(logos_dir):
        for f in os.listdir(logos_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                logo_paths.append(os.path.join(logos_dir, f))
    
    # Add legacy logos if they exist and aren't duplicates
    for legacy in [assets_logo]:
        if os.path.exists(legacy):
            # Check if already in list by filename
            legacy_name = os.path.basename(legacy)
            if not any(os.path.basename(p) == legacy_name for p in logo_paths):
                logo_paths.append(legacy)
    
    return sorted(logo_paths)


def _load_logo_state() -> dict:
    """Load logo rotation state from file."""
    try:
        if os.path.exists(LOGO_STATE_FILE):
            with open(LOGO_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        core.logging.log_event(f"Error loading logo state: {e}", "WARNING")
    return {"current_index": 0}


def _save_logo_state(state: dict):
    """Save logo rotation state to file."""
    try:
        with open(LOGO_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        core.logging.log_event(f"Error saving logo state: {e}", "WARNING")


def get_next_logo() -> Optional[Image.Image]:
    """
    Returns the next logo from the rotating pool.
    Cycles through all available logos sequentially.
    """
    logo_paths = get_logo_paths()
    if not logo_paths:
        core.logging.log_event("No logos found in assets/logos directory", "WARNING")
        return None
    
    state = _load_logo_state()
    current_index = state.get("current_index", 0) % len(logo_paths)
    
    logo_path = logo_paths[current_index]
    
    # Update state for next call
    state["current_index"] = (current_index + 1) % len(logo_paths)
    state["last_logo"] = os.path.basename(logo_path)
    _save_logo_state(state)
    
    try:
        logo = Image.open(logo_path).convert("RGBA")
        core.logging.log_event(f"Using logo: {os.path.basename(logo_path)}", "INFO")
        return logo
    except Exception as e:
        core.logging.log_event(f"Error loading logo {logo_path}: {e}", "ERROR")
        return None


# --- Energy Analysis ---
def analyze_image_energy(image: Image.Image) -> np.ndarray:
    """
    Compute energy map using Sobel gradients.
    Low energy = flat/uniform areas ideal for watermarks.
    High energy = edges, details, busy areas to avoid.
    
    Args:
        image: PIL Image to analyze
        
    Returns:
        2D numpy array of energy values (same size as image)
    """
    # Convert to grayscale numpy array
    gray = np.array(image.convert('L'), dtype=np.float64)
    
    # Compute Sobel gradients
    dx = ndimage.sobel(gray, axis=1)  # Horizontal gradient
    dy = ndimage.sobel(gray, axis=0)  # Vertical gradient
    
    # Energy is magnitude of gradient
    energy = np.sqrt(dx**2 + dy**2)
    
    # Normalize to 0-1 range
    if energy.max() > 0:
        energy = energy / energy.max()
    
    return energy


def compute_local_orientation(energy_map: np.ndarray, region: Tuple[int, int, int, int]) -> float:
    """
    Analyze local gradients to determine optimal text rotation.
    Aligns text with dominant gradient direction for natural integration.
    
    Args:
        energy_map: Full image energy map
        region: (x, y, width, height) of the watermark region
        
    Returns:
        Optimal rotation angle in degrees (-30 to 30)
    """
    x, y, w, h = region
    
    # Extract region
    region_energy = energy_map[y:y+h, x:x+w]
    
    # Compute gradient direction in the region
    dy, dx = np.gradient(region_energy)
    
    # Average gradient direction
    avg_dx = np.mean(dx)
    avg_dy = np.mean(dy)
    
    # Calculate angle (clamped to reasonable range)
    if abs(avg_dx) > 0.001 or abs(avg_dy) > 0.001:
        angle = math.degrees(math.atan2(avg_dy, avg_dx))
        # Clamp to -30 to 30 degrees for subtle effect
        angle = max(-30, min(30, angle * 0.3))
    else:
        angle = 0
    
    return angle


def compute_local_skew(energy_map: np.ndarray, region: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Compute skew factors based on energy gradient flow.
    
    Args:
        energy_map: Full image energy map
        region: (x, y, width, height) of the watermark region
        
    Returns:
        (skew_x, skew_y) factors for affine transformation
    """
    x, y, w, h = region
    region_energy = energy_map[y:y+h, x:x+w]
    
    # Compute gradient
    dy, dx = np.gradient(region_energy)
    
    # Skew based on gradient variation
    skew_x = np.std(dx) * 0.1  # Subtle horizontal skew
    skew_y = np.std(dy) * 0.1  # Subtle vertical skew
    
    # Clamp to reasonable values
    skew_x = max(-0.15, min(0.15, skew_x))
    skew_y = max(-0.15, min(0.15, skew_y))
    
    return skew_x, skew_y


def find_optimal_watermark_position(
    image: Image.Image, 
    watermark_size: Tuple[int, int],
    energy_map: Optional[np.ndarray] = None
) -> Tuple[int, int, float, float]:
    """
    Find position with lowest energy that fits the watermark.
    
    Args:
        image: Source image
        watermark_size: (width, height) of watermark
        energy_map: Pre-computed energy map (optional)
        
    Returns:
        (x, y, angle, scale) for optimal placement
    """
    if energy_map is None:
        energy_map = analyze_image_energy(image)
    
    img_w, img_h = image.size
    wm_w, wm_h = watermark_size
    
    # Define candidate regions (corners and edges, avoiding center)
    margin = WATERMARK_MARGIN
    candidates = [
        # Bottom-right (most common)
        (img_w - wm_w - margin, img_h - wm_h - margin),
        # Bottom-left
        (margin, img_h - wm_h - margin),
        # Top-right
        (img_w - wm_w - margin, margin),
        # Top-left
        (margin, margin),
        # Bottom-center
        ((img_w - wm_w) // 2, img_h - wm_h - margin),
        # Right-center
        (img_w - wm_w - margin, (img_h - wm_h) // 2),
    ]
    
    best_pos = candidates[0]
    best_energy = float('inf')
    
    for x, y in candidates:
        # Ensure within bounds
        x = max(0, min(x, img_w - wm_w))
        y = max(0, min(y, img_h - wm_h))
        
        # Calculate average energy in this region
        region = energy_map[y:y+wm_h, x:x+wm_w]
        avg_energy = np.mean(region)
        
        if avg_energy < best_energy:
            best_energy = avg_energy
            best_pos = (x, y)
    
    x, y = best_pos
    
    # Compute optimal rotation based on local gradients
    angle = compute_local_orientation(energy_map, (x, y, wm_w, wm_h))
    
    # Scale based on energy variance (lower variance = can be larger)
    region = energy_map[y:y+wm_h, x:x+wm_w]
    variance = np.var(region)
    # Scale inversely with variance (busy areas get smaller watermark)
    scale = 1.0 - (variance * 0.3)
    scale = max(0.7, min(1.0, scale))
    
    return x, y, angle, scale


# --- Watermark Creation ---
def create_text_watermark(
    text: str,
    font_size: int,
    opacity: float = 1.0,
    rotation: float = 0,
    skew: Tuple[float, float] = (0, 0)
) -> Image.Image:
    """
    Creates a text watermark image.
    
    Args:
        text: Text to render
        font_size: Font size in points
        opacity: Opacity (0-1)
        rotation: Rotation angle
        skew: Skew factors
        
    Returns:
        RGBA image of the text
    """
    # Find font
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\impact.ttf",
    ]
    
    font = None
    for path in font_paths:
        try:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        except Exception:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Calculate size
    dummy_img = Image.new('RGBA', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Create image with padding for rotation/skew
    padding = int(max(text_width, text_height) * 0.5)
    img_size = (text_width + padding * 2, text_height + padding * 2)
    
    watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Draw text centered
    x = (img_size[0] - text_width) // 2
    y = (img_size[1] - text_height) // 2
    
    # White text with optional opacity
    alpha = int(255 * opacity)
    text_color = (255, 255, 255, alpha)
    
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Cropping box to trim empty space after transforms
    bbox = [x, y, x + text_width, y + text_height]
    
    # Apply rotation
    if abs(rotation) > 0.5:
        watermark = watermark.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    
    # Apply skew
    if abs(skew[0]) > 0.01 or abs(skew[1]) > 0.01:
        w, h = watermark.size
        skew_x, skew_y = skew
        coeffs = (1, skew_x, -skew_x * h / 2,
                  skew_y, 1, -skew_y * w / 2)
        watermark = watermark.transform(
            (w, h), 
            Image.Transform.AFFINE, 
            coeffs,
            resample=Image.Resampling.BICUBIC
        )
    
    return watermark.crop(watermark.getbbox())


def mask_energy_region(energy_map: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Mask a region in the energy map by setting it to maximum energy.
    Used to prevent overlapping watermarks.
    
    Args:
        energy_map: Source energy map
        region: (x, y, w, h) to mask
        
    Returns:
        Modified energy map
    """
    result = energy_map.copy()
    x, y, w, h = region
    
    # Ensure bounds
    h_map, w_map = result.shape
    x = max(0, min(x, w_map))
    y = max(0, min(y, h_map))
    w = min(w, w_map - x)
    h = min(h, h_map - y)
    
    # Set to max energy (1.0) plus a buffer to discourage placing right next to it
    result[y:y+h, x:x+w] = 1.0
    
    return result


# --- Main Entry Point ---
def apply_watermark(
    image: Image.Image, 
    opacity: float = WATERMARK_OPACITY,
    force_position: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Main entry point - applies intelligent split watermarks (Logo + Hidden Text).
    
    Args:
        image: Source PIL Image
        opacity: Watermark opacity (0-1) for the main logo
        force_position: Optional forced (x, y) position for the LOGO
        
    Returns:
        Image with watermarks applied
    """
    try:
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        img_w, img_h = image.size
        energy_map = analyze_image_energy(image)
        
        # --- 1. Place Main Logo ---
        logo = get_next_logo()
        if logo:
            # Calculate optimal logo size
            max_logo_size = int(min(img_w, img_h) * WATERMARK_MAX_SIZE_RATIO)
            min_logo_size = int(min(img_w, img_h) * WATERMARK_MIN_SIZE_RATIO)
            target_logo_size = max(min_logo_size, min(max_logo_size, 150)) # Cap at 150px
            
            # Resize logo maintaining aspect ratio
            aspect = logo.width / logo.height
            if aspect > 1:
                logo_w = target_logo_size
                logo_h = int(target_logo_size / aspect)
            else:
                logo_h = target_logo_size
                logo_w = int(target_logo_size * aspect)
            
            logo_resized = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
            
            # Find position
            if force_position:
                lx, ly = force_position
                l_angle = 0
                l_scale = 1.0
            else:
                lx, ly, l_angle, l_scale = find_optimal_watermark_position(
                    image, 
                    (logo_w, logo_h),
                    energy_map
                )
            
            # Apply opacity to logo
            if opacity < 1.0:
                alpha = logo_resized.split()[3] if 'A' in logo_resized.getbands() else logo_resized.convert('RGBA').split()[3]
                alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                logo_resized.putalpha(alpha)
            
            # Paste logo
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Adjust position to be safe within image
            lx = max(WATERMARK_MARGIN, min(lx, img_w - logo_w - WATERMARK_MARGIN))
            ly = max(WATERMARK_MARGIN, min(ly, img_h - logo_h - WATERMARK_MARGIN))
            
            image.paste(logo_resized, (lx, ly), logo_resized)
            
            core.logging.log_event(f"Logo placed at ({lx}, {ly})", "INFO")
            
            # Update energy map to mask the logo area + margin
            mask_margin = 20
            energy_map = mask_energy_region(
                energy_map, 
                (lx - mask_margin, ly - mask_margin, logo_w + mask_margin*2, logo_h + mask_margin*2)
            )
            
        # --- 2. Place Hidden "@e-v-l-o-v-e.bsky.social" Text ---
        # Fixed small size ~8pt (approx 11px at 96dpi, but let's say 12px for visibility)
        # We'll scale it slightly with image size but keep it small
        base_font_size = 12
        font_scale = min(img_w, img_h) / 1024
        font_size = int(max(12, base_font_size * font_scale))
        
        # Generate text image to get dimensions
        text_img_temp = create_text_watermark(WATERMARK_TEXT, font_size)
        tw, th = text_img_temp.size
        
        # Find position for text using updated energy map
        tx, ty, t_angle, t_scale = find_optimal_watermark_position(
            image, 
            (tw, th),
            energy_map
        )
        
        # Compute skew for text to make it blend organically
        t_skew = compute_local_skew(energy_map, (tx, ty, tw, th))
        
        # Create final text watermark
        hidden_text = create_text_watermark(
            WATERMARK_TEXT, 
            font_size, 
            opacity=0.15, # Very subtle
            rotation=t_angle,
            skew=t_skew
        )
        
        # Paste text
        tx = max(WATERMARK_MARGIN, min(tx, img_w - hidden_text.width - WATERMARK_MARGIN))
        ty = max(WATERMARK_MARGIN, min(ty, img_h - hidden_text.height - WATERMARK_MARGIN))
        
        image.paste(hidden_text, (tx, ty), hidden_text)
        
        core.logging.log_event(f"Hidden '{WATERMARK_TEXT}' placed at ({tx}, {ty})", "INFO")
        
        return image.convert('RGB')
        
    except Exception as e:
        core.logging.log_event(f"Error applying watermark: {e}", "ERROR")
        return image


# --- Utility Functions ---
def generate_energy_visualization(image: Image.Image) -> Image.Image:
    """
    Generate a visualization of the image energy map.
    Useful for debugging and understanding placement decisions.
    
    Args:
        image: Source image
        
    Returns:
        Grayscale image showing energy levels
    """
    energy = analyze_image_energy(image)
    
    # Convert to 8-bit grayscale
    energy_8bit = (energy * 255).astype(np.uint8)
    
    return Image.fromarray(energy_8bit, mode='L')


def get_watermark_stats() -> dict:
    """Get statistics about the watermark system."""
    logo_paths = get_logo_paths()
    state = _load_logo_state()
    
    return {
        "total_logos": len(logo_paths),
        "current_index": state.get("current_index", 0),
        "last_logo_used": state.get("last_logo", None),
        "logos": [os.path.basename(p) for p in logo_paths]
    }
