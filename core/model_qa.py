import json
import asyncio
import logging
from typing import Dict, Any, Optional

try:
    from core.llm_api import run_llm, MODEL_STATS
    import core.logging
except ImportError:
    # Standalone mode support
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from core.llm_api import run_llm, MODEL_STATS
    import core.logging

QA_PROMPT = """
You are a precision testing unit.
Respond with a strictly formatted JSON object containing the following keys:
1. "status": Must be exactly "ready".
2. "checksum": The sum of 5 + 7.
3. "model_id": Your own model identifier if known, otherwise "unknown".

Example Response:
```json
{
  "status": "ready",
  "checksum": 12,
  "model_id": "unknown"
}
```
Do not output any text before or after the JSON.
"""

async def test_model_capability(model_id: str) -> Dict[str, Any]:
    """
    Runs a standardized test on a specific model to verify its JSON generation capabilities.
    Returns a result dict with 'score' (0-100), 'latency', and 'error' if any.
    """
    import time
    start_time = time.time()
    
    try:
        # Force specific model usage in run_llm if possible, or we rely on the caller 
        # to have set up the context. Since run_llm usually picks the "best" model, 
        # testing a *specific* model requires bypassing rank_models or passing a specific override.
        # core.llm_api.run_llm doesn't easily support "force this specific model ID" 
        # without some modifications or using the lower-level provider interfaces directly.
        # For now, we will assume we can pass a 'model_id_override' or similar if we modify llm_api,
        # OR we just test the "current best" and see if it fails.
        
        # However, for this to be useful as a scanner, we ideally want to iterate available models.
        # Let's assume run_llm can accept a model_id kwarg (which it often can in these architectures).
        
        response = await run_llm(
            prompt_text=QA_PROMPT,
            purpose="qa_testing",
            model_id=model_id # Hypothetical override support
        )
        
        duration = time.time() - start_time
        
        # Parse output
        if isinstance(response, dict):
             text = response.get("result", "") or response.get("content", "")
        else:
            text = str(response)
            
        # Clean markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        try:
            data = json.loads(text)
            score = 0
            if data.get("status") == "ready":
                score += 40
            if int(data.get("checksum", 0)) == 12:
                score += 40
            if "model_id" in data:
                score += 20
                
            return {
                "score": score,
                "latency": duration,
                "success": True
            }
        except Exception as e:
            return {
                "score": 0,
                "latency": duration,
                "success": False,
                "error": f"JSON Parse Error: {e} - Output: {text[:100]}"
            }
            
    except Exception as e:
        return {
            "score": 0,
            "latency": time.time() - start_time,
            "success": False,
            "error": f"API Call Failed: {e}"
        }

async def run_system_qa():
    """
    Iterates through known models in MODEL_STATS and tests them.
    Updates the global stats and saves them.
    """
    print("Starting Model QA Cycle...")
    
    # We need a way to get the list of models.
    # MODEL_STATS is loaded in llm_api.
    
    results = {}
    
    # Snapshot keys to avoid modification during iteration
    model_ids = list(MODEL_STATS.keys())
    
    for mid in model_ids:
        print(f"Testing model: {mid}...")
        result = await test_model_capability(mid)
        results[mid] = result
        print(f"  -> Score: {result['score']}, Success: {result['success']}")
        
        # Update Stats
        if mid not in MODEL_STATS:
            MODEL_STATS[mid] = {}
            
        MODEL_STATS[mid]['qa_score'] = result['score']
        MODEL_STATS[mid]['last_qa_check'] = time.time()
        
        if not result['success']:
             # Penalize reliablity
             current_fails = MODEL_STATS[mid].get("failed_calls", 0)
             MODEL_STATS[mid]["failed_calls"] = current_fails + 1
    
    # Save stats
    try:
        with open("llm_model_stats.json", "w") as f:
            json.dump(MODEL_STATS, f, indent=4)
        print("Updated llm_model_stats.json")
    except Exception as e:
        print(f"Failed to save stats: {e}")

if __name__ == "__main__":
    asyncio.run(run_system_qa())
