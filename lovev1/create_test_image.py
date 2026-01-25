from rich.text import Text
from rich.console import Console
from core.art_utils import save_ansi_art
import os
import sys

# Ensure current directory is in path (it usually is for scripts)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    console = Console()
    
    # Create a Rich Text object with multiple colors and styles
    text = Text()
    text.append("ANSI Art Color Rendering Test\n", style="bold underline white")
    text.append("-----------------------------\n", style="white")
    text.append("Red Text ", style="bold red")
    text.append("Green Text ", style="bold green")
    text.append("Blue Text\n", style="bold blue")
    text.append("Yellow on Blue Background\n", style="bold yellow on blue")
    text.append("Magenta Italic\n", style="magenta italic")
    text.append("Cyan Strikethrough\n", style="cyan strike")
    text.append("-----------------------------\n", style="white")
    text.append("L.O.V.E. System Active", style="bold green blink")
    
    print("Generating colored test image...")
    ansi, svg, png = save_ansi_art(text, "color_test_image", output_dir="art_test")
    
    print(f"Generated files in 'art_test' directory:")
    print(f"ANSI: {ansi}")
    print(f"SVG: {svg}")
    print(f"PNG: {png}")
    
    if png and os.path.exists(png):
        print("SUCCESS: Colored PNG created.")
    else:
        print("FAILURE: PNG creation failed.")

if __name__ == "__main__":
    create_test_image()
