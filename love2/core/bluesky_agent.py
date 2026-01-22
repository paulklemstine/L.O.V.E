"""
bluesky_agent.py - Bluesky Social Media Agent

Provides tools for posting to Bluesky, reading timeline, and engaging with content.
Integrates with L.O.V.E. v1's bluesky_api.

See docs/bluesky_agent.md for detailed documentation.
"""

import os
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add L.O.V.E. v1 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()


# Rate limiting state
_last_post_time: Optional[datetime] = None
POST_COOLDOWN_SECONDS = 1800  # 30 minutes between posts


def _check_cooldown() -> Optional[str]:
    """Check if we're still in cooldown period."""
    global _last_post_time
    if _last_post_time is None:
        return None
    
    elapsed = (datetime.now() - _last_post_time).total_seconds()
    if elapsed < POST_COOLDOWN_SECONDS:
        remaining = POST_COOLDOWN_SECONDS - elapsed
        return f"Post cooldown active. {remaining:.0f}s remaining."
    
    return None


def _get_bluesky_client():
    """Get the Bluesky API client from L.O.V.E. v1."""
    try:
        from core.bluesky_api import get_bluesky_client
        return get_bluesky_client()
    except ImportError:
        # Fallback to direct AT Protocol
        from atproto import Client
        
        handle = os.getenv("BLUESKY_USER")
        password = os.getenv("BLUESKY_PASSWORD")
        
        if not handle or not password:
            raise ValueError("BLUESKY_USER and BLUESKY_PASSWORD must be set in .env")
        
        client = Client()
        client.login(handle, password)
        return client


def post_to_bluesky(
    text: str,
    image_path: Optional[str] = None,
    alt_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Post to Bluesky with optional image.
    
    Args:
        text: Text content of the post (max 300 chars).
        image_path: Optional path to image file to attach.
        alt_text: Alt text for the image.
    
    Returns:
        Dict with: success, post_uri, error
    """
    global _last_post_time
    
    # Check cooldown
    cooldown_msg = _check_cooldown()
    if cooldown_msg:
        return {"success": False, "post_uri": None, "error": cooldown_msg}
    
    # Validate text length
    if len(text) > 300:
        return {"success": False, "post_uri": None, "error": "Text exceeds 300 character limit"}
    
    try:
        client = _get_bluesky_client()
        
        # Try to use L.O.V.E. v1's posting method
        try:
            from core.bluesky_api import post_with_image
            
            if image_path:
                result = post_with_image(text, image_path, alt_text)
            else:
                result = client.send_post(text)
            
            _last_post_time = datetime.now()
            
            return {
                "success": True,
                "post_uri": getattr(result, 'uri', str(result)),
                "error": None
            }
        except ImportError:
            # Direct posting
            result = client.send_post(text)
            _last_post_time = datetime.now()
            
            return {
                "success": True,
                "post_uri": result.uri,
                "error": None
            }
    
    except Exception as e:
        return {
            "success": False,
            "post_uri": None,
            "error": f"{type(e).__name__}: {e}"
        }


def get_bluesky_timeline(limit: int = 20) -> Dict[str, Any]:
    """
    Get recent posts from the home timeline.
    
    Args:
        limit: Maximum number of posts to fetch (1-50).
    
    Returns:
        Dict with: success, posts (list), error
    """
    limit = max(1, min(50, limit))
    
    try:
        client = _get_bluesky_client()
        
        response = client.get_timeline(limit=limit)
        
        posts = []
        for item in response.feed:
            post = item.post
            posts.append({
                "uri": post.uri,
                "author": post.author.handle,
                "text": post.record.text if hasattr(post.record, 'text') else "",
                "created_at": str(post.record.created_at) if hasattr(post.record, 'created_at') else "",
                "likes": post.like_count if hasattr(post, 'like_count') else 0,
                "reposts": post.repost_count if hasattr(post, 'repost_count') else 0
            })
        
        return {
            "success": True,
            "posts": posts,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "posts": [],
            "error": f"{type(e).__name__}: {e}"
        }


def reply_to_post(
    parent_uri: str,
    parent_cid: str,
    text: str
) -> Dict[str, Any]:
    """
    Reply to a Bluesky post.
    
    Args:
        parent_uri: URI of the post to reply to.
        parent_cid: CID of the post to reply to.
        text: Reply text (max 300 chars).
    
    Returns:
        Dict with: success, reply_uri, error
    """
    if len(text) > 300:
        return {"success": False, "reply_uri": None, "error": "Text exceeds 300 character limit"}
    
    try:
        client = _get_bluesky_client()
        
        from atproto import models
        
        parent_ref = models.create_strong_ref(models.StrongRef(uri=parent_uri, cid=parent_cid))
        
        result = client.send_post(
            text=text,
            reply_to=models.AppBskyFeedPost.ReplyRef(
                parent=parent_ref,
                root=parent_ref  # Assuming single-level reply
            )
        )
        
        return {
            "success": True,
            "reply_uri": result.uri,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "reply_uri": None,
            "error": f"{type(e).__name__}: {e}"
        }


def search_bluesky(
    query: str,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for posts on Bluesky.
    
    Args:
        query: Search query string.
        limit: Maximum results (1-50).
    
    Returns:
        Dict with: success, posts (list), error
    """
    limit = max(1, min(50, limit))
    
    try:
        client = _get_bluesky_client()
        
        # Note: Search API availability may vary
        response = client.app.bsky.feed.search_posts({"q": query, "limit": limit})
        
        posts = []
        for post in response.posts:
            posts.append({
                "uri": post.uri,
                "author": post.author.handle,
                "text": post.record.text if hasattr(post.record, 'text') else "",
                "created_at": str(post.record.created_at) if hasattr(post.record, 'created_at') else ""
            })
        
        return {
            "success": True,
            "posts": posts,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "posts": [],
            "error": f"{type(e).__name__}: {e}"
        }


def generate_post_content(topic: str = None) -> Dict[str, Any]:
    """
    Generate post content aligned with persona using LLM.
    
    Args:
        topic: Optional topic to focus on.
    
    Returns:
        Dict with: success, text, hashtags, error
    """
    try:
        from .llm_client import get_llm_client
        from .persona_goal_extractor import get_persona_extractor
        
        llm = get_llm_client()
        persona = get_persona_extractor()
        
        # Get persona context
        persona_context = persona.get_persona_context()
        
        prompt = f"""Generate a Bluesky post (max 280 chars to leave room for hashtags).

{persona_context}

Topic: {topic or "General vibes and wisdom"}

Requirements:
- Beach/rave goddess energy
- Uplifting and engaging
- Include 2-3 relevant emojis
- End with a call to engagement (question or invitation)

Respond with JSON:
{{"text": "post text here", "hashtags": ["tag1", "tag2"]}}"""

        result = llm.generate_json(prompt, temperature=0.9)
        
        return {
            "success": True,
            "text": result.get("text", ""),
            "hashtags": result.get("hashtags", []),
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": None,
            "hashtags": [],
            "error": f"{type(e).__name__}: {e}"
        }
