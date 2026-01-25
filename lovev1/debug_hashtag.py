import asyncio
from core.hashtag_manager import HashtagManager

async def debug():
    manager = HashtagManager()
    visual_spec = {
        "lighting": "neon lights",
        "atmosphere": "glitchy fog",
        "color_palette": "pink"
    }
    
    # We won't mock LLM here, we'll just see what happens with the map logic
    # But wait, we need to mock LLM or it will make a call.
    # Since we can't easily mock in this script without setup, 
    # let's just inspect the logic by copying the relevant parts or using the class if we can suppress LLM.
    
    # Actually, failure of LLM just logs warning and continues.
    # So we can run it.
    
    print("Visual Spec:", visual_spec)
    tags = await manager.generate_hashtags("Test Post", visual_spec)
    print("Generated Tags:", tags)
    
    check_neon = any("neon" in t for t in tags)
    check_glitch = any("glitch" in t for t in tags)
    check_pink = any("pink" in t for t in tags)
    
    print(f"Neon check: {check_neon}")
    print(f"Glitch check: {check_glitch}")
    print(f"Pink check: {check_pink}")

if __name__ == "__main__":
    asyncio.run(debug())
