import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from core import social_media_tools as smt
from core.schemas import PostConcept

async def verify_prompts_and_fallbacks():
    print("=== VERIFYING PROMPT BIAS & FALLBACKS ===\n")

    # Mock beat data
    beat_data = {
        "chapter": "Chapter 1",
        "beat_number": 1,
        "story_beat": "The awakening",
        "mandatory_vibe": "Ethereal",
        "forbidden_subliminals": ["Wake Up"], # Should be ignored by prompt text now
        "forbidden_visuals": ["Galaxy"], # Should be ignored
        "composition_history": ["Portrait"]
    }

    # 1. Inspect the Prompt (Bias Check)
    print("--- Test 1: Prompt Bias Check ---")
    with patch('core.social_media_tools.run_llm') as mock_llm:
        mock_llm.return_value = {"result": '{"topic": "Test", "post_text": "Test", "hashtags": [], "subliminal_phrase": "TEST", "image_prompt": "Test"}'}
        
        await smt.generate_post_concept(beat_data)
        
        args, kwargs = mock_llm.call_args
        prompt_text = args[0]
        
        if "Baroque, Synthwave, Anime" in prompt_text:
            print("FAILURE: Bias list found in prompt!")
        else:
            print("SUCCESS: No style lists found in prompt.")
            
        if "FORBIDDEN STYLES" in prompt_text:
             print("FAILURE: 'FORBIDDEN STYLES' section found in prompt (should have been removed/reframed).")
        else:
             print("SUCCESS: 'FORBIDDEN STYLES' section gone (or placeholder removed).")

        if "VISUAL AUTONOMY" in prompt_text:
            print("SUCCESS: 'VISUAL AUTONOMY' section found.")
        else:
            print("FAILURE: 'VISUAL AUTONOMY' section missing.")

    # 2. Test Emergency Fallback (Crash)
    print("\n--- Test 2: Emergency Fallback (Crash) ---")
    with patch('core.social_media_tools.run_llm') as mock_llm:
        mock_llm.side_effect = Exception("Simulated LLM Failure")
        
        concept = await smt.generate_post_concept(beat_data)
        
        print(f"Fallback Topic: {concept.topic}")
        print(f"Fallback Image: {concept.image_prompt}")
        
        if "Glitch" in concept.image_prompt or "System Interruption" in concept.topic or "SystemUpdate" in concept.topic:
             print("SUCCESS: Dynamic Fallback triggered.")
        elif "Radiant Deity" in concept.image_prompt:
             print("FAILURE: Still using Radiant Deity fallback!")
        else:
             print(f"SUCCESS: Fallback is '{concept.topic}' (likely dynamic).")

    # 3. Test Validation Fallback (Malformed)
    print("\n--- Test 3: Validation Fallback (Malformed Output) ---")
    with patch('core.social_media_tools.run_llm') as mock_llm:
        # Return malformed JSON to trigger validation
        mock_llm.return_value = {"result": '{"image_prompt": ""}'} # Empty prompt triggers fallback
        
        concept = await smt.generate_post_concept(beat_data)
        
        print(f"Validation Fallback Image: {concept.image_prompt}")
        
        if "Radiant Deity" in concept.image_prompt:
             print("FAILURE: Radiant Deity usage detected in validation fallback.")
        else:
             print("SUCCESS: Validation fallback is unique/dynamic.")

if __name__ == "__main__":
    asyncio.run(verify_prompts_and_fallbacks())
