
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import get_llm_client

async def use_client(i):
    client = get_llm_client()
    print(f"Run {i}: Getting client...")
    # This just triggers _get_async_client logic
    try:
        ac = await client._get_async_client()
        print(f"Run {i}: Client loop is {getattr(client, '_client_loop', 'unknown')}")
        print(f"Run {i}: Current loop is {asyncio.get_running_loop()}")
        print(f"Run {i}: Client is closed? {ac.is_closed}")
        return True
    except Exception as e:
        print(f"Run {i}: FAILED: {e}")
        return False

def main():
    print("--- Run 1 ---")
    asyncio.run(use_client(1))
    
    print("\n--- Run 2 (New Loop) ---")
    try:
        asyncio.run(use_client(2))
        print("Run 2 Success!")
    except Exception as e:
        print(f"Run 2 Crashed: {e}")

if __name__ == "__main__":
    main()
