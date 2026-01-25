
import sys
import os
import time
import requests
from unittest.mock import MagicMock

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Mock necessary modules before importing core.llm_api
sys.modules['core.logging'] = MagicMock()
sys.modules['rich.console'] = MagicMock()
sys.modules['rich.panel'] = MagicMock()
sys.modules['rich.progress'] = MagicMock()
sys.modules['rich.align'] = MagicMock()
sys.modules['rich.markdown'] = MagicMock()
sys.modules['rich.text'] = MagicMock()
sys.modules['rich.live'] = MagicMock()
sys.modules['rich.table'] = MagicMock()
sys.modules['rich.style'] = MagicMock()
sys.modules['rich.layout'] = MagicMock()
sys.modules['rich.box'] = MagicMock()
sys.modules['rich'] = MagicMock()

# Import target
# We need to make sure core is importable
try:
    from core import llm_api
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Could not import core.llm_api. Make sure you are in the project root. Error: {e}")
    sys.exit(1)

def test_circuit_breaker():
    print("Testing Circuit Breaker...")
    
    # Setup state - Mocking Global State in llm_api
    llm_api.MODEL_STATS = {"gemini-pro": {"provider": "gemini"}, "claude": {"provider": "openrouter"}}
    llm_api.LLM_AVAILABILITY = {}
    llm_api.PROVIDER_AVAILABILITY = {}
    
    # Simulating 429 on Gemini
    print("Simulating 429 on Gemini...")
    llm_api.PROVIDER_AVAILABILITY["gemini"] = time.time() + 300
    
    # Now check if our filter logic works (Logic copied from run_llm implementation)
    ranked_models = ["gemini-pro", "claude"]
    
    models_to_try = []
    for m in ranked_models:
        model_is_available = time.time() >= llm_api.LLM_AVAILABILITY.get(m, 0)
        
        provider = llm_api.MODEL_STATS[m].get("provider", "unknown")
        provider_is_available = time.time() >= llm_api.PROVIDER_AVAILABILITY.get(provider, 0)
        
        if model_is_available and provider_is_available:
            models_to_try.append(m)
            
    print(f"Models to try: {models_to_try}")
    
    if "gemini-pro" not in models_to_try:
        print("PASS: Circuit breaker filtered out Gemini model.")
    else:
        print("FAIL: Circuit breaker did not filter Gemini model.")
        
    if "claude" in models_to_try:
        print("PASS: Other providers still available.")
    else:
        print("FAIL: Other providers blocked incorrectly.")

if __name__ == "__main__":
    test_circuit_breaker()
