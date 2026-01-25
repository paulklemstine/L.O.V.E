import asyncio
import sys
import os

# Ensure we can import from core/root
sys.path.append(os.getcwd())

from display import create_blessing_panel
from rich.text import Text
from rich.console import Console

async def main():
    print("Starting reproduction...")
    
    # Simulate the inputs
    blessing_text = "Generate additional resources through innovative partnerships and investments."
    
    # Simulate ansi_art as a Rich Text object (as returned by generate_llm_art)
    # Using some dummy ANSI text
    ansi_content = "\033[31mHello\033[0m \033[32mWorld\033[0m"
    ansi_art = Text.from_ansi(ansi_content)
    
    width = 80
    
    print(f"Calling create_blessing_panel with width={width}...")
    try:
        panel = await create_blessing_panel(blessing_text, ansi_art=ansi_art, width=width)
        print("Panel created successfully!")
        
        # Try to render it to verify it doesn't crash on render
        console = Console()
        console.print(panel)
        print("Panel rendered successfully!")
        
    except Exception as e:
        print(f"Caught exception: '{e}'")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
