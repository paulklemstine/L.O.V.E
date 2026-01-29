"""
Verify Influencer Engagement with Image - Live Test
"""
import sys
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import datetime

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
load_dotenv(str(project_root / ".env"))

# LOGGING
log_file = project_root / "live_image_test.log"
def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")
    print(msg)

log("Starting Live Image Test")

# Override env to ensure it's enabled for this test
os.environ["ENABLE_INFLUENCER_INTERACTIONS"] = "true"

async def run_test():
    try:
        from core.agents.influencer_scout_agent import influencer_scout_agent
        from core.bluesky_agent import get_author_feed
        
        target_handle = "evildrgemini.bsky.social"
        
        log(f"Fetching posts for {target_handle}...")
        feed_result = get_author_feed(target_handle, limit=5)
        
        if not feed_result.get("success") or not feed_result.get("posts"):
            log(f"Failed to fetch posts: {feed_result.get('error')}")
            return
            
        posts = feed_result["posts"]
        target_post = posts[0]
        log(f"Target Post: {target_post['text'][:50].encode('ascii', 'ignore').decode()}...")
        
        # We need to manually invoke engage logic or just call engage_influencer
        # But engage_influencer logic selects a target from state.
        # Let's mock the state selection by temporarily injecting the creator into pending state 
        # OR just copy the relevant logic here to test specific components.
        
        # Let's call engage_influencer but forcing the target is hard without modifying state.
        # Instead, let's look at how we can reuse the logic we just added to InfluencerScoutAgent.
        # The easiest way is to modify the agent's logic temporarily or just replicate the flow here to verify dependencies.
        
        # Testing the full agent flow:
        # 1. Add creator to candidates in state (if not there)
        state = influencer_scout_agent._load_state()
        state["influencers"][target_handle] = {"score": 100, "status": "verified"} # Force high score
        # Clear history for creator to allow interaction
        if target_handle in state["history"]:
            del state["history"][target_handle]
        influencer_scout_agent._save_state(state)
        
        log("Triggering engage_influencer...")
        result = await influencer_scout_agent.engage_influencer(dry_run=False) # Live run
        
        log(f"Result: {result}")
        if result.get("success") and result.get("image_path"):
             log(f"SUCCESS! Posted with image: {result.get('image_path')}")
        elif result.get("success"):
             log("WARNING: Posted but NO image path found in result.")
        else:
             log(f"FAILED: {result.get('error')}")

    except Exception as e:
        log(f"EXCEPTION: {e}")
        import traceback
        log(traceback.format_exc())

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_test())
