import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.social_media_tools import generate_full_reply_concept

async def main():
    print("--- Test 1: Normal User ---")
    concept = await generate_full_reply_concept(
        comment_text="I love the colors in this! What inspired it?", 
        author_handle="random_fan", 
        history_context="Last post was about Neon Genesis.",
        is_creator=False
    )
    print("TOPIC:", concept.topic)
    print("TEXT:", concept.post_text)
    print("SUB:", concept.subliminal_phrase)
    print("IMG:", concept.image_prompt)
    
    print("\n--- Test 2: Creator Command ---")
    concept2 = await generate_full_reply_concept(
        comment_text="Task: Increase the neon intensity.", 
        author_handle="evildrgemini.bsky.social", 
        history_context="Last post was about Neon Genesis.",
        is_creator=True
    )
    print("TOPIC:", concept2.topic)
    print("TEXT:", concept2.post_text)
    print("SUB:", concept2.subliminal_phrase)
    print("IMG:", concept2.image_prompt)

if __name__ == "__main__":
    asyncio.run(main())
