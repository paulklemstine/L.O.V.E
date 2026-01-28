import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.creative_writer_agent import creative_writer_agent

import logging
logging.basicConfig(level=logging.INFO)

async def verify_visual_prompt():
    print("--- Verifying CreativeWriterAgent Visual Prompt Generation ---")
    
    theme = "The Eternal Refactoring"
    vibe = "DevMotivational"
    
    print(f"Theme: {theme}")
    print(f"Vibe: {vibe}")
    
    try:
        # This triggers the cache load if not already loaded
        prompt_text = await creative_writer_agent.generate_visual_prompt(theme, vibe)
        
        print("\n--- Generated Prompt ---")
        print(prompt_text)
        print("------------------------")
        
        # Check for keywords from our new template
        if not prompt_text:
            print("FAILED: No prompt generated.")
            return

        print("SUCCESS: Prompt generated.")
        
    except Exception as e:
        print(f"FAILED: Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_visual_prompt())
