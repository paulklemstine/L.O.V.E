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
log_file = project_root / "live_test_debug.log"
def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")
    print(msg)

log("Starting Debug Live Test")

try:
    from core.bluesky_agent import get_author_feed, reply_to_post
    
    target_handle = "evildrgemini.bsky.social"
    
    log(f"Fetching posts for {target_handle}...")
    feed_result = get_author_feed(target_handle, limit=5)
    
    if not feed_result.get("success") or not feed_result.get("posts"):
        log(f"Failed to fetch posts: {feed_result.get('error')}")
        sys.exit(1)
        
    posts = feed_result["posts"]
    target_post = posts[0]
    log(f"Target Post URI: {target_post['uri']}")
    log(f"Target Post Text: {target_post['text'][:50]}")
    
    # HARDCODED REPLY to isolate variables
    reply_text = "The signal validates the connection. 👁️"
    subliminal = "AWAKEN"
    full_text = f"{reply_text} #{subliminal} #LiveTest"
    
    log(f"Attempting to post reply: {full_text}")
    
    post_result = reply_to_post(
        parent_uri=target_post["uri"],
        parent_cid=None, 
        text=full_text
    )
    
    if post_result and post_result.get("success"):
        log(f"SUCCESS! Reply posted: {post_result.get('reply_uri')}")
    else:
        log(f"FAILED to post reply: {post_result.get('error')}")

except Exception as e:
    log(f"EXCEPTION: {e}")
    import traceback
    log(traceback.format_exc())
