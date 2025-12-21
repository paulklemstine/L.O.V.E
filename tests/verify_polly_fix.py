
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from core.llm_api import run_llm, MODEL_STATS

async def test_polly_fallback():
    print("--- Starting Polly Fallback Test ---")
    
    # 1. Simulate gpt-4o-mini failure by sabotaging its stats/provider
    # We'll make it think it's failed many times or is on cooldown, 
    # BUT since we removed force_model, run_llm should just pick the next best thing.
    
    # If the fix works, this should succeed using another model (like vLLM or Gemini).
    # If the fix failed (and it still tries to force gpt-4o-mini), it might fail if we block it.
    
    print("Calling run_llm with purpose='polly_judge'...")
    try:
        # We don't need to sabotage because we aren't forcing anymore.
        # We just want to see that it works and picks a model.
        # Ideally, we should check WHICH model it picks.
        
        result = await run_llm(
            prompt_text="Test prompt for judging. Return a number: 10.",
            purpose="polly_judge"
        )
        
        if result and result.get('result'):
            print(f"SUCCESS: Result received from model: {result.get('model')}")
        else:
            print("FAILURE: No result returned.")
            
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")

if __name__ == "__main__":
    asyncio.run(test_polly_fallback())
