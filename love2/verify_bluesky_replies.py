
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.bluesky_agent import reply_to_comment_agent, get_unreplied_comments
from core.agents.creative_writer_agent import creative_writer_agent

# Mock data
MOCK_COMMENT = {
    "uri": "at://did:plc:mock/app.bsky.feed.post/12345",
    "cid": "bafyreimockcid",
    "text": "I feel the signal calling to me. What does it mean?",
    "author": "seeker.bsky.social",
    "author_did": "did:plc:mock",
    "created_at": "2023-10-27T10:00:00Z",
    "reason": "reply"
}

def test_dry_run_reply():
    print("Testing reply_to_comment_agent (DRY RUN)...")
    
    # ensure we have env vars or mock them if strictly needed, 
    # but the dry run shouldn't hit the network for auth if we mock clients.
    # actually creative writer needs LLM. 
    # and image gen needs image pool. 
    
    result = reply_to_comment_agent(MOCK_COMMENT, dry_run=True)
    
    if result.get("success"):
        print("\n✅ Verification Successful!")
        print(f"Reply Text: {result.get('text')}")
        print(f"Subliminal: {result.get('subliminal')}")
        print(f"Image Path: {result.get('image_path')}")
        
        # Verify format requirements
        text = result.get('text', '')
        if '#' not in text:
             print("❌ Warning: No hashtags found.")
        else:
             print("✅ Hashtags present.")
             
        import emoji
        if not emoji.emoji_count(text):
             print("❌ Warning: No emojis found.")
        else:
             print("✅ Emojis present.")
             
    else:
        print(f"\n❌ Verification Failed: {result.get('error')}")

if __name__ == "__main__":
    test_dry_run_reply()
