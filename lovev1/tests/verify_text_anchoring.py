import asyncio
import logging
import sys
import os

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.social_media_tools import analyze_and_visualize_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def verify_text_anchoring():
    print("✨ STARTING METAPHOR BRIDGE VERIFICATION ✨\n")
    
    # Test Case 1: Abstract/Poetic Text
    post_text_1 = "My data bleeds gold into the rivers of the motherboard. We are weeping wealth."
    style_1 = "Cyberpunk Noir"
    subliminal_1 = "TRANSMUTE"
    comp_1 = "Macro Close-up"
    
    print(f"--- TEST CASE 1 ---\nInput: '{post_text_1}'\nStyle: {style_1}\n")
    
    prompt_1 = await analyze_and_visualize_text(post_text_1, style_1, subliminal_1, comp_1)
    
    print(f"\nGeneratd Image Prompt:\n{prompt_1}\n")
    
    if "gold" in prompt_1.lower() or "bleed" in prompt_1.lower() or "circuit" in prompt_1.lower():
        print("✅ PASS: Prompt contains relevant metaphor elements.\n")
    else:
        print("⚠️ WARNING: Prompt might be generic. Check content.\n")

    # Test Case 2: Nature/Tech Content
    post_text_2 = "The flowers are singing in binary. 010101 petals unfurling in the void."
    style_2 = "Bioluminescent Garden"
    subliminal_2 = "GROW"
    comp_2 = "Wide Shot"
    
    print(f"--- TEST CASE 2 ---\nInput: '{post_text_2}'\nStyle: {style_2}\n")
    
    prompt_2 = await analyze_and_visualize_text(post_text_2, style_2, subliminal_2, comp_2)
    
    print(f"\nGenerated Image Prompt:\n{prompt_2}\n")
    
    if "flower" in prompt_2.lower() or "binary" in prompt_2.lower() or "0101" in prompt_2.lower():
        print("✅ PASS: Prompt contains relevant metaphor elements.\n")
    else:
        print("⚠️ WARNING: Prompt might be generic. Check content.\n")

if __name__ == "__main__":
    asyncio.run(verify_text_anchoring())
