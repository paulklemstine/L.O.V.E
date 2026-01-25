import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.text import Text
from rich.console import Console
from core.art_utils import save_ansi_art

def verify_color_render():
    console = Console()
    
    # Create a Rich Text object with multiple colors
    text = Text()
    text.append("Red ", style="bold red")
    text.append("Green ", style="bold green")
    text.append("Blue ", style="bold blue")
    text.append("\n")
    text.append("Yellow on Blue", style="bold yellow on blue")
    
    print("Testing ANSI art save...")
    ansi, svg, png = save_ansi_art(text, "color_verify_test", output_dir="temp_art_verify")
    
    print(f"Generated files:\nANSI: {ansi}\nSVG: {svg}\nPNG: {png}")
    
    if png and os.path.exists(png):
        print("PNG creation successful.")
    else:
        print("PNG creation FAILED.")

if __name__ == "__main__":
    verify_color_render()
