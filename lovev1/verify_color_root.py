from rich.text import Text
from rich.console import Console
from core.art_utils import save_ansi_art
import os
import sys

# Ensure current directory is in path (it usually is for scripts)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_color_render():
    console = Console()
    
    # Create a Rich Text object with multiple colors
    text = Text()
    text.append("Red ", style="bold red")
    text.append("Green ", style="bold green")
    text.append("Blue ", style="bold blue")
    text.append("\n")
    text.append("Yellow on Blue", style="bold yellow on blue")
    text.append("\n")
    text.append("ANSI Art Color Test", style="magenta italic")
    
    print("Testing ANSI art save from root...")
    ansi, svg, png = save_ansi_art(text, "color_verify_root", output_dir="temp_art_verify")
    
    print(f"Generated files:\nANSI: {ansi}\nSVG: {svg}\nPNG: {png}")
    
    if png and os.path.exists(png):
        print("PNG creation successful.")
    else:
        print("PNG creation FAILED.")

if __name__ == "__main__":
    verify_color_render()
