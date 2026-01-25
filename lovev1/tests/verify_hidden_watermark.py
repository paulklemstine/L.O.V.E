import os
import sys
import numpy as np
from PIL import Image

# Add core to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.watermark import apply_watermark

def create_test_image(width=1024, height=1024):
    """Create a gradient image to test energy map."""
    # Create gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # Simple pattern: Low energy in corners, high energy in center
    z = np.sin(xv * 10) * np.cos(yv * 10)
    
    # Normalize to 0-255
    data = ((z + 1) / 2 * 255).astype(np.uint8)
    
    return Image.fromarray(data).convert("RGB")

def main():
    print("Creating test image...")
    img = create_test_image()
    
    print("Applying watermark...")
    result = apply_watermark(img)
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "test_hidden_watermark.png")
    result.save(output_path)
    
    print(f"Saved result to {output_path}")
    print("Please manually verify that there is a Logo and a tiny 'l.o.v.e' text in different spots.")

if __name__ == "__main__":
    main()
