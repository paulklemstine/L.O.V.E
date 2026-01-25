
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
print(f"DEBUG: sys.path: {sys.path}")

try:
    import core
    print(f"DEBUG: core imported from {core.__file__}")
    import core.agents
    print(f"DEBUG: core.agents imported from {core.agents.__file__}")
except Exception as e:
    print(f"DEBUG: Import error: {e}")

from core.agents.response_optimizer import ResponseOptimizer

async def test_zero_successful_calls():
    optimizer = ResponseOptimizer(tool_registry={})
    
    # Mock stats with 0 successful calls
    stats = {
        "model_a": {
            "total_time_spent": 1000,
            "successful_calls": 0,
            "total_tokens_generated": 500
        },
        "model_b": {
            "total_time_spent": 500,
            "successful_calls": 10,
            "total_tokens_generated": 1000
        }
    }
    
    try:
        issues = await optimizer._identify_inefficiencies(stats)
        print("Successfully analyzed stats without error.")
        if issues:
            print("Issues found:", issues)
        else:
            print("No issues found (expected for empty/zero stats).")
            
    except ZeroDivisionError:
        print("FAIL: ZeroDivisionError caught!")
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_zero_successful_calls())
