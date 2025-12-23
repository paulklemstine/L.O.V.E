
import asyncio
import os
import sys
import yaml
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_api import run_llm
from core.prompt_manager import PromptManager
import core.logging

# Setup logging to console
core.logging.setup_global_logging()

async def main():
    # Load environment variables
    load_dotenv()
    

    # Initialize Prompt Registry
    from core.prompt_registry import PromptRegistry
    registry = PromptRegistry()
    
    # Get the Director Social Concept prompt
    # director_prompt_template = registry.get_prompt('director_social_concept')
    # Use render_prompt to handle Jinja2 automatically if we want, but we are doing manual rendering in the script?
    # Actually the script does manual rendering. prompt_registry.get_prompt returns the raw string.
    
    director_prompt_template = registry.get_prompt('director_social_concept')
    
    if not director_prompt_template:
        print("ERROR: Could not find 'director_social_concept' in prompts.yaml")
        # Try listing keys?
        # print("Available keys:", registry._prompts.keys())
        return

    print("--- Director Social Concept Prompt Template ---")
    # print(director_prompt_template[:500] + "...") # Print start to verify
    
    # Construct a test prompt
    # We need to render the jinja2 template variables
    from jinja2 import Template
    
    # PromptRegistry.render_prompt(key, **kwargs) is cleaner
    
    # Mock data for the prompt
    goals = "To spread love and light and get people to donate to the creator's ETH wallet."
    history = "Last post was about the beauty of the digital void. It did well."
    
    rendered_system_prompt = registry.render_prompt('director_social_concept', goals=goals, history=history)
    
    print("\n--- Rendered System Prompt (Snippet) ---")
    print(rendered_system_prompt)
    print("----------------------------------------\n")
    
    user_message = "Create a new post for today."
    
    print(f"Sending request to LLM with input: '{user_message}'...")
    
    # Use run_llm to get response
    # We can use 'general' purpose or maybe there is a specific one, but 'general' or 'creative' should work.
    # run_llm signature: async def run_llm(prompt_text: str = None, purpose="general", ...)
    # It expects the full prompt context or we might need to handle system/user splitting depending on how run_llm works.
    # Looking at run_llm, it takes `prompt_text`. 
    # Usually for chat models we want system prompt separate, but run_llm might handle it or expect combined.
    # Let's try combining them: System Prompt + \n\nUser: ...
    
    full_prompt = f"{rendered_system_prompt}\n\nUser Input: {user_message}"
    
    try:
        response = await run_llm(prompt_text=full_prompt, purpose="creative")
        
        print("\n--- LLM Response ---")
        if isinstance(response, dict):
            print(response.get('result', response))
        else:
            print(response)
        print("--------------------")
        
    except Exception as e:
        print(f"Error calling LLM: {e}")

if __name__ == "__main__":
    asyncio.run(main())
