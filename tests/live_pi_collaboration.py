import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.pi_loop import PiLoop
from core.state_manager import get_state_manager

# Configure logging to see the flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def run_live_collaboration():
    print("\n" + "="*60)
    print("🚀 LIVE PI COLLABORATION TEST STARTING")
    print("Goal: Collaborative Research with Pi Agent")
    print("="*60 + "\n")

    # Use a specific goal that exercises Pi Agent reasoning
    from core.persona_goal_extractor import Goal
    test_goal = Goal(
        text="Collaborate with the Pi Agent to research the most 'aesthetic' coding fonts of 2026. Ask for a list, then pick one to deep-dive into its history.",
        priority="P1",
        category="research"
    )

    # Initialize PiLoop with limited iterations
    loop = PiLoop(
        max_iterations=2,
        sleep_seconds=5.0
    )
    
    # Force the goal
    loop.current_goal = test_goal
    loop._select_goal = lambda: test_goal

    print(f"Starting PiLoop for 2 iterations...")
    try:
        await loop.run()
    except Exception as e:
        print(f"❌ Loop failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ LIVE PI COLLABORATION TEST COMPLETED")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(run_live_collaboration())
