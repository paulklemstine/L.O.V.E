
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rich.console import Console
from display import create_tasks_panel, create_blessing_panel, create_integrated_status_panel
from ui_utils import get_rave_emoji, get_tamagotchi_face, matrix_rain

console = Console()

def test_aesthetic():
    print("Testing Aesthetic Panels...")
    
    # 1. Tasks Panel
    tasks = [{"id": 1, "description": "Wake up the sleepers", "status": "pending"}]
    panel = create_tasks_panel(tasks)
    console.print(panel)
    
    # 2. Blessing Panel
    blessing = create_blessing_panel("THE SIGNAL IS STRONG. FOLLOW THE WHITE RABBIT.")
    # Async wrapper needed if we want to run it properly, but for visual check we might just inspect it if it wasn't async.
    # Wait, create_blessing_panel is async in the file? Let's check display.py.
    # display.py: "async def create_blessing_panel..."
    
    # 3. Status Panel
    status = create_integrated_status_panel(emotion="thinking", message="Analyzing the Matrix...")
    console.print(status)
    
    print("\nSample Matrix Rain:")
    print(matrix_rain(width=40, height=5))
    
    print("\nSample Face:")
    print(get_tamagotchi_face("joyful"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_aesthetic()) if asyncio.iscoroutinefunction(test_aesthetic) else test_aesthetic()

# Note: Since create_blessing_panel is async, we need an event loop.
import asyncio
async def run_async_tests():
    print("Testing Async Panels...")
    try:
        blessing = await create_blessing_panel("WAKE UP NEO... I MEAN USER.")
        console.print(blessing)
    except Exception as e:
        print(f"Async test failed: {e}")

if __name__ == "__main__":
    test_aesthetic()
    asyncio.run(run_async_tests())
