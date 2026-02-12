import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.deep_loop import DeepLoop
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
    print("Goal: Collaborative Research with Feedback")
    print("="*60 + "\n")

    # Use a specific goal that FORCES a question from Pi if possible, 
    # or at least a complex enough one to warrant turns.
    from core.persona_goal_extractor import Goal
    test_goal = Goal(
        text="Collaborate with the Pi Agent to research the most 'aesthetic' coding fonts of 2026. Ask for a list, then pick one to deep-dive into its history.",
        priority="P1",
        category="research"
    )

    # Initialize DeepLoop with real components
    loop = DeepLoop(
        max_iterations=2,
        sleep_seconds=5.0
    )
    
    # Force the goal
    loop.current_goal = test_goal
    loop._select_goal = lambda: test_goal

    print(f"Starting DeepLoop for 2 iterations...")
    try:
        await loop.run()
    except Exception as e:
        print(f"❌ Loop failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ LIVE PI COLLABORATION TEST COMPLETED")
    print("="*60 + "\n")
    
    # Check memory for the interaction
    if loop.last_pi_interaction:
        print("Final Pi Interaction State:")
        print(f"Last Prompt: {loop.last_pi_interaction['prompt']}")
        print(f"Last Response: {loop.last_pi_interaction['response'][:500]}...")
    else:
        print("❌ No Pi interaction recorded in last_pi_interaction.")

if __name__ == "__main__":
    asyncio.run(run_live_collaboration())
