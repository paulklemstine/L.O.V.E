import asyncio
import sys
import os
import core.logging

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tools_legacy import evolve

async def main():
    print("Triggering evolution demo...")
    core.logging.log_event("Triggering evolution demo script", "INFO")
    
    # Run the evolve tool without a goal to trigger the auto-evolution "Baby Steps" protocol
    result = await evolve(goal=None)
    
    print(f"Result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
