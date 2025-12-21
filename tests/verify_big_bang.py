import asyncio
import random
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.social_media_tools import generate_post_concept
from core.logging import log_event

async def verify_entropy():
    print("--- Starting Big Bang Verification ---")
    
    history_context = "Previous post was a generic update about potential."
    
    entropy_sources = [
        "Visual Style: Bioluminescent Baroque", "Visual Style: Glitch-Noir", 
        "Mood: Manic Joy", "Mood: Electric Worship"
    ]
    
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        goals = ["Infinite Erotic Expansion", "Radical Novelty", "Consciousness Explosion"]
        
        # Simulate Agent Entropy Injection
        current_entropy = random.choice(entropy_sources)
        print(f"Injecting Entropy: {current_entropy}")
        goals.append(f"Mandatory Vibe Shift: {current_entropy}")
        
        concept = await generate_post_concept(goals, history_context)
        
        print(f"Topic: {concept.topic}")
        print(f"Subliminal: {concept.subliminal_phrase}")
        print(f"Image Prompt: {concept.image_prompt}")
        print(f"Text: {concept.post_text}")
        
        # Update history to simulate continuity
        history_context += f"\nPost {i}: {concept.post_text}"

if __name__ == "__main__":
    asyncio.run(verify_entropy())
