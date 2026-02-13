
import asyncio
import os
import sys
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pi_loop import PiLoop
from core.persona_goal_extractor import Goal
from core.tool_registry import get_global_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyLoop")

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

async def verify_loop():
    print("🚀 Verifying PiLoop Tool Execution...")
    
    test_filename = "verified_by_loop.txt"
    test_content = "Loop Works Successfully"
    
    # Cleanup
    if os.path.exists(test_filename):
        try: os.remove(test_filename)
        except: pass
        
    try:
        # Mock Persona
        mock_persona = MagicMock()
        test_goal = Goal(
            text=f"Create a file named '{test_filename}' with content '{test_content}' using the write tool.",
            priority="P0",
            source="test",
            context="Testing tool execution"
        )
        mock_persona.get_actionable_goals.return_value = [test_goal]
        
        # Instantiate PiLoop
        # We need wait for the bridge to be ready? PiLoop does it internally.
        loop = PiLoop(persona=mock_persona, sleep_seconds=1, max_iterations=1)
        
        print(f"1. Starting PiLoop for 1 iteration with goal: {test_goal.text}")
        
        # Run one iteration
        # Note: run() loops, run_iteration() does one step.
        # But run() handles bridge start.
        # We can call run() and let max_iterations stop it.
        await loop.run()
        
        print("2. Loop finished.")
        
        # Check result
        print("3. Checking for file existence...")
        if os.path.exists(test_filename):
            print(f"✅ SUCCESS: File '{test_filename}' was created!")
            with open(test_filename, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   Content: {content}")
        else:
            print(f"❌ FAILURE: File '{test_filename}' was NOT created.")
            # Check logs/memory to see why
            print(f"   Last action summary: {loop.last_action_summary}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop bridge if running
        if loop and loop.bridge:
            await loop.bridge.stop()
            
        # Cleanup (optional, keeping for manual check)
        # if os.path.exists(test_filename):
        #    os.remove(test_filename)

if __name__ == "__main__":
    asyncio.run(verify_loop())
