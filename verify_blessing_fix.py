import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich_gradient.gradient import Gradient
from display import create_blessing_panel
import sys

# Mocking RAVE_COLORS just in case
RAVE_COLORS = ["#FF00FF", "#00FFFF", "#FFFF00"]

async def verify_fix():
    console = Console()
    
    # 1. Test case WITH ANSI art - Should return Panel (not Gradient)
    ansi_art_str = "\033[31mRED ART\033[0m \033[34mBLUE ART\033[0m"
    print("\n--- Testing Blessing Panel WITH ANSI Art ---")
    result_with_art = await create_blessing_panel("Here is your art", ansi_art=ansi_art_str)
    
    is_panel = isinstance(result_with_art, Panel)
    is_gradient = isinstance(result_with_art, Gradient)
    
    print(f"Result type: {type(result_with_art)}")
    if is_panel and not is_gradient:
        print("PASS: Result is a Panel (Gradient skipped to preserve colors).")
        console.print(result_with_art)
    else:
        print("FAIL: Result is NOT a plain Panel (Gradient likely still applied).")
        try:
            console.print(result_with_art)
        except:
            pass
        sys.exit(1)

    # 2. Test case WITHOUT ANSI art - Should return Gradient
    print("\n--- Testing Blessing Panel WITHOUT ANSI Art ---")
    result_no_art = await create_blessing_panel("Just a message")
    
    is_gradient_normal = isinstance(result_no_art, Gradient)
    
    print(f"Result type: {type(result_no_art)}")
    type_str = str(type(result_no_art))
    if "Gradient" in type_str or is_gradient_normal:
        print("PASS: Result is a Gradient type (Standard styling applied).")

        try:
            console.print(result_no_art)
        except:
            pass # Gradient might fail in some envs if not fully mocked, but type check is key
    else:
        print("FAIL: Result should be a Gradient when no art is present.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify_fix())
