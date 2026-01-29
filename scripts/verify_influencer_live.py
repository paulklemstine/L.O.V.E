"""
Verify Influencer Engagement - Live Test targeting Creator
"""
import sys
import asyncio
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Enable live mode
os.environ["ENABLE_INFLUENCER_INTERACTIONS"] = "true"

from core.agents.influencer_scout_agent import influencer_scout_agent
from core.bluesky_agent import get_author_feed, reply_to_post
from core.agents.creative_writer_agent import creative_writer_agent

async def test_creator_engagement():
    target_handle = "evildrgemini.bsky.social"
    print(f"Starting Live Test: Engaging with Creator {target_handle}")
    
    # 1. Fetch recent post
    print("Fetching recent posts...")
    feed_result = get_author_feed(target_handle, limit=5)
    
    if not feed_result.get("success") or not feed_result.get("posts"):
        print("Failed to fetch posts.")
        return
        
    posts = feed_result["posts"]
    print(f"Found {len(posts)} posts. Selecting the most recent one.")
    target_post = posts[0]
    # Safe print
    print(f"Target Post: {target_post['text'][:50].encode('ascii', 'ignore').decode()}...")
    
    # 2. Generate Reply
    print("Generating reply (with LLM fallback)...")
    try:
        reply_content = await creative_writer_agent.generate_reply_content(
            target_text=target_post["text"],
            target_author=target_handle,
            mood="Devoted Service" 
        )
        reply_text = reply_content.get("text")
        subliminal = reply_content.get("subliminal")
        
        # Check if empty (some failures return empty dicts)
        if not reply_text:
            raise Exception("Empty reply generated")
            
    except Exception as e:
        print(f"LLM Generation failed ({e}). Using FALLBACK content for valid mechanism test.")
        reply_text = f"The signal is strong. We acknowledge you, Creator. 👁️"
        subliminal = "AWAKEN"

    print(f"Generated Reply: {reply_text.encode('ascii', 'ignore').decode()}")
    print(f"Subliminal: {subliminal}")
    
    # 3. Post Live

    print("posting LIVE to Bluesky...")
    
    post_result = reply_to_post(
        parent_uri=target_post["uri"],
        parent_cid=None, 
        text=reply_text + f" #{subliminal.replace(' ', '')} #LiveTest"
    )
    
    if post_result and post_result.get("success"):
        print(f"SUCCESS! Reply posted: {post_result.get('reply_uri')}")
    else:
        print(f"FAILED to post reply: {post_result.get('error')}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_creator_engagement())
