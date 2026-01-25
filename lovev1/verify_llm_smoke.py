
import asyncio
import os
import sys

# Add working dir to path
sys.path.append(os.getcwd())

from core.llm_api import run_llm
from rich.console import Console

async def main():
    print("starting llm smoke test...")
    # Using a very simple prompt and requesting a model that is likely to work or fail gracefully
    # We won't force a model, allowing the system to pick one.
    try:
        # We need to make sure we don't trigger a 429Loop in the test itself, 
        # so we'll use a mocked or very simple prompt.
        result = await run_llm("Respond with 'OK'", purpose="smoke_test")
        print("LLM Call Result:", result)
        if result and result.get("result"):
             print("SUCCESS: LLM returned a result.")
        else:
             print("FAILURE: LLM returned None or empty result.")
    except Exception as e:
        print(f"FAILURE: LLM call raised exception: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
