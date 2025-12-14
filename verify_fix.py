
import asyncio
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

async def test_polly_fix():
    print("Testing Polly Fix...")
    
    # Mock core.polly.run_llm to return {'result': None}
    with patch('core.polly.run_llm') as mock_llm:
        mock_llm.return_value = {'result': None}
        
        from core.polly import PollyOptimizer
        
        optimizer = PollyOptimizer()
        # Mock registry to avoid file I/O
        optimizer.registry = MagicMock()
        optimizer.registry.get_prompt.return_value = "Test Prompt"
        
        print("Running optimize_prompt...")
        result = await optimizer.optimize_prompt("test_key")
        
        print(f"Result: {result}")
        
        if result is None:
            print("SUCCESS: optimize_prompt handled None result gracefully.")
        else:
            print(f"FAILURE: optimize_prompt returned unexpected value: {result}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_polly_fix())
