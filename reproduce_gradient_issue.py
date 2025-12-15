from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich_gradient.gradient import Gradient
import sys

# Mocking RAVE_COLORS just in case, though likely not needed for this minimal repro
RAVE_COLORS = ["#FF00FF", "#00FFFF", "#FFFF00"]

def test_gradient_overwrite():
    console = Console()
    
    # Create some text with specific colors (ANSI art simulation)
    # colored_text = Text("This should be RED", style="red")
    # colored_text.append(" and this BLUE", style="blue")
    
    # Simulating ANSI art more directly
    ansi_art_str = "\033[31mRED ART\033[0m \033[34mBLUE ART\033[0m"
    art_renderable = Text.from_ansi(ansi_art_str)
    
    print("--- Original Art Renderable ---")
    console.print(art_renderable)
    
    # Create a Panel containing this art
    panel = Panel(
        art_renderable,
        title="Test Panel",
        border_style="green",
        padding=(1, 2)
    )
    
    print("\n--- Panel (Normal) ---")
    console.print(panel)
    
    # Wrap in Gradient
    gradient_panel = Gradient(panel, colors=["hot_pink", "bright_cyan", "yellow1"])
    
    print("\n--- Panel (Gradient Wrapped) - EXPECT BROKEN COLORS ---")
    try:
        console.print(gradient_panel)
    except Exception as e:
        print(f"Error printing gradient panel: {e}")

if __name__ == "__main__":
    test_gradient_overwrite()
