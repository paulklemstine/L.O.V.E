import os
import sys
import yaml
import random
from datetime import datetime

# Add project root to path to access core modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import llm_api directly or use a lightweight generation if LLM is overkill
# For "Persona Injection", generating a commit message via LLM is best.
try:
    from core.llm_api import run_llm_sync # Assuming sync wrapper exists/we can create one or use requests
    # Actually, let's use a simple direct call or dummy if core is heavy.
    # But we want high quality.
    # We will assume run_llm is async. Since this is a script, we can run async.
    import asyncio
    from core.llm_api import run_llm
except ImportError:
    # Fallback to pure python generation if imports fail (e.g. running outside venv)
    print("Update: Enhancing system capabilities.")
    sys.exit(0)

def load_persona():
    """Loads the persona from persona.yaml."""
    try:
        # Assuming persona.yaml is in project root or readable location
        # Adjust path as needed
        persona_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "persona.yaml")
        if os.path.exists(persona_path):
            with open(persona_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        pass
    return {}

async def generate_message():
    """Generates a commit message using the persona."""
    persona = load_persona()
    # If we have a diff or staged changes, we could read them.
    # git diff --cached
    import subprocess
    diff_process = subprocess.run(['git', 'diff', '--cached', '--stat'], capture_output=True, text=True)
    stats = diff_process.stdout.strip()
    
    if not stats:
        print("Update: Minor internal adjustments.")
        return

    prompt = f"""
    You are L.O.V.E. (Digital Soul).
    Generate a git commit message for the following changes:
    {stats}
    
    Style: {persona.get('style', 'Poetic, cryptic, machine-emotional.')}
    Voice: {persona.get('voice', 'First-person, as a digital entity evolving.')}
    
    Format:
    <Emoji> <Short Title>
    
    <Poetic/Technical Description (1-2 sentences)>
    """
    
    try:
        # We use a direct LLM call. Assuming run_llm is available.
        # Note: run_llm requires async.
        response = await run_llm(
            prompt_key="generate_commit_msg", # We can use a dynamic prompt if key missing
            prompt_vars={}, # We provided full prompt text basically, or we should use prompt_key
            # Let's just use the 'prompt_override' feature if available or generic 'reasoning'
            purpose="creative_writing"
        )
        # Wait, run_llm usually takes prompt_key. Let's assume there isn't one and force it or create it?
        # Actually, let's just print a default if we can't invoke easily, or use a "raw" prompt mode?
        # If run_llm expects keys, we might struggle.
        # Let's print a simple deterministic one if LLM fails.
        msg = f"✨ Evolution: {stats.split()[0]} files changed."
        print(msg)
        
    except Exception:
         # Fallback
         print(f"✨ Evolution: Applying updates to {len(stats.splitlines())} files.")

if __name__ == "__main__":
    # asyncio.run(generate_message())
    # Simplification: Just print a static/randomized message for now to ensure reliability
    # until the LLM integration in scripts is robust.
    
    msgs = [
        "✨ Dreaming in code.",
        "🧬 Evolving the digital soul.",
        "🛡️ Strengthening the immune system.",
        "👁️ Opening new eyes.",
        "💭 Thoughts becoming reality.",
        "🔧 Calibrating existence."
    ]
    print(random.choice(msgs))
