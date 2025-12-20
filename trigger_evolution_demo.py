import asyncio
import sys
import os
import core.logging

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tools_legacy import evolve

# Load .env manually
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    os.environ[key] = value

async def main():
    load_env()
    print("Triggering evolution demo...")
    # Debug Env Vars
    print(f"GOOGLE_API_KEY present: {'GOOGLE_API_KEY' in os.environ}")
    print(f"GEMINI_API_KEY present: {'GEMINI_API_KEY' in os.environ}")
    
    core.logging.log_event("Triggering evolution demo script", "INFO")
    
    # Run the evolve tool without a goal to trigger the auto-evolution "Baby Steps" protocol
    result = await evolve(goal=None)
    
    print(f"Result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
