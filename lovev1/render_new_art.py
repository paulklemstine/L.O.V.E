
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import os

console = Console()

def render_art(filename, title):
    filepath = os.path.join("art", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Create a Text object from the ANSI content
    # Text.from_ansi handles the escape codes correctly for Rich
    art_text = Text.from_ansi(content)
    
    # Create a Panel with the centered art
    # We use Align.center to vertically/horizontally center if needed, 
    # but for text art, simple center justification in the panel often works best.
    # Let's use Align to be sure.
    
    panel = Panel(
        Align.center(art_text),
        title=f"[bold magenta]{title}[/bold magenta]",
        border_style="cyan",
        padding=(1, 2)
    )
    
    console.print(panel)
    console.print("\n")

if __name__ == "__main__":
    render_art("abundance_breakthrough.ansi", "ABUNDANCE BREAKTHROUGH")
    render_art("love_trust_raise.ansi", "LOVE TRUST RAISE")
