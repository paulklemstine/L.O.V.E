"""
bluesky_agent.py - Bluesky Social Media Agent

Provides tools for posting to Bluesky, reading timeline, and engaging with content.
Integrates with L.O.V.E. v2's local bluesky_api.

See docs/bluesky_agent.md for detailed documentation.
"""

import os
import sys
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import emoji

load_dotenv()


# Rate limiting state
POST_COOLDOWN_SECONDS = 300  # 5 minutes between posts
GENERATION_COOLDOWN_SECONDS = 300  # 5 minutes between image generations

# Rate limiting state
_last_post_time: Optional[datetime] = None
_last_gen_time: Optional[datetime] = None


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
    """Get the Bluesky API client from local module."""
    try:
        from .bluesky_api import get_bluesky_client
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
        # Try to use local posting method
        try:
            from .bluesky_api import post_to_bluesky_with_image
            
            if image_path:
                print(f"[BlueskyAgent] Posting with image using local API: {image_path}")
                # Load image
                from PIL import Image
                image = Image.open(image_path)
                result = post_to_bluesky_with_image(text, image)
            else:
                client = _get_bluesky_client()
                result = client.send_post(text)
            
            _last_post_time = datetime.now()
            
            return {
                "success": True,
                "post_uri": getattr(result, 'uri', str(result)),
                "error": None
            }
        except ImportError as ie:
            print(f"[BlueskyAgent] Could not import post_to_bluesky_with_image: {ie}")
            # Direct posting fallback
            client = _get_bluesky_client()
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



def _validate_post_content(text: str, hashtags: List[str], subliminal_phrase: str) -> List[str]:
    """
    Validate generated content against QA rules.
    
    Checks:
    1. Total length (text + hashtags) <= 300
    2. At least 1 emoji present
    3. Subliminal phrase NOT in open text
    4. At least 1 hashtag
    """
    errors = []
    
    # 1. Length check
    # Estimate hashtag length: #tag + space
    tags_len = sum(len(t) + 1 for t in hashtags) 
    total_len = len(text) + tags_len
    if total_len > 300:
        errors.append(f"Content too long ({total_len}/300 chars)")
        
    # 2. Emoji check
    if not emoji.emoji_count(text):
        errors.append("No emojis found in text")
        
    # 3. Subliminal check
    if subliminal_phrase and subliminal_phrase.lower() in text.lower():
        errors.append(f"Subliminal phrase '{subliminal_phrase}' exposed in text")
        
    if not hashtags:
        errors.append("No hashtags generated")
    
    if errors:
        print(f"[BlueskyAgent] ⚠️ Post validation failed: {'; '.join(errors)}")
        
    return errors


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
    parent_uri: str = None,
    parent_cid: str = None,
    text: str = None,
    **kwargs
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
    if not parent_uri or not parent_cid or not text:
        return {"success": False, "reply_uri": None, "error": "Missing required arguments: parent_uri, parent_cid, text"}

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


def generate_post_content(topic: str = None, auto_post: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Generate post content using the sophisticated Story Pipeline (v1 vibe).
    
    Pipeline:
    1. StoryManager: Determines the current "Beat" (Narrative + Vibe + Visuals)
    2. CreativeWriterAgent: Generates "Micro-Story" and "Subliminal Phrase"
    3. Hashtag Generation: Creates psychological/manipulative tags
    4. Image Generation: Creates image matching the beat's visual style
    
    Args:
        topic: Optional override (though StoryManager usually drives this).
        auto_post: If True, continuously posts to Bluesky.
    
    Returns:
        Dict with post details.
    """
    # Check cooldown first if auto-posting
    if auto_post:
        cooldown_msg = _check_cooldown()
        if cooldown_msg:
            return {
                "success": False, 
                "error": f"Cannot auto-post: {cooldown_msg}",
                "text": None,
                "image_path": None
            }

    try:
        import asyncio
        from .story_manager import story_manager
        from .agents.creative_writer_agent import creative_writer_agent
        from .logger import log_event
        
        # Configure StoryManager to use 'state/story_state.json'
        state_dir = Path(__file__).parent.parent / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        story_manager.state_file = str(state_dir / "story_state.json")
        # Reload state with new path if needed, or just rely on next save/load
        story_manager.state = story_manager._load_state()

        # 1. Get the Story Beat (The "Seed")
        beat_data = story_manager.get_next_beat()
        log_event(f"Story Beat Generated: {beat_data.get('story_beat')}", "INFO")
        
        # Override topic if provided, otherwise use the beat
        theme = topic if topic else beat_data.get("story_beat")
        vibe = beat_data.get("mandatory_vibe", "Ethereal")
        
        # 2. Generate Content via CreativeWriter (The "Voice")
        # We need to run async methods in this sync function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        content_task = creative_writer_agent.write_micro_story(
            theme=theme, 
            mood=vibe,
            memory_context=beat_data.get("previous_beat", "")
        )
        content_result = loop.run_until_complete(content_task)
        
        text = content_result.get("story", "")
        subliminal = content_result.get("subliminal", "")
        
        # 3. Generate Hashtags
        hashtag_task = creative_writer_agent.generate_manipulative_hashtags(
            topic=theme,
            count=3
        )
        hashtags = loop.run_until_complete(hashtag_task)
        
        # 4. Generate Image (The "Visuals")
        image_path = None
        global _last_gen_time
        gen_cooldown_active = False
        if _last_gen_time:
            elapsed = (datetime.now() - _last_gen_time).total_seconds()
            if elapsed < GENERATION_COOLDOWN_SECONDS:
                gen_cooldown_active = True
                print("[BlueskyAgent] Image cooldown active. Skipping image.")

        if not gen_cooldown_active:
            try:
                from .image_generation_pool import generate_image_with_pool
                from .watermark import apply_watermark
                
                # Construct Image Prompt from Beat Data
                # Allows the StoryManager to control visual consistency
                # For now, we'll ask the LLM to merge them if we had a VisualAgent, 
                # but let's use a simple strategy:
                
                # Use the vibe to select a preset or generate a prompt
                # But since we ported prompt_manager, let's use a simple construct for now
                # In v1 this was complex. Here we simplify:
                visual_prompt = f"{vibe} aesthetic, {theme}, cinematic lighting, 8k, masterpiece"
                
                if subliminal:
                    # Pass subliminal to image generator if it supports text overlay/embedding
                    pass

                print(f"[BlueskyAgent] Generating image: {visual_prompt}")
                
                image_result = loop.run_until_complete(
                    generate_image_with_pool(
                        prompt=visual_prompt, 
                        text_content=subliminal 
                    )
                )
                
                image = image_result[0] if isinstance(image_result, tuple) else image_result
                
                if image:
                    image = apply_watermark(image)
                    filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    save_dir = state_dir / "images"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    image_path = str(save_dir / filename)
                    image.save(image_path)
                    print(f"[BlueskyAgent] Image saved to {image_path}")
                    _last_gen_time = datetime.now()
                    
                    # Record visual style usage
                    story_manager.record_post(subliminal, visual_prompt)
                    
            except Exception as e:
                print(f"[BlueskyAgent] Image generation failed: {e}")
                # Still record the post text part
                story_manager.record_post(subliminal, "")

        # 5. Format Output
        full_text = f"{text}\n\n{' '.join(hashtags)}"
        
        # 6. Auto-Post
        post_result = None
        if auto_post:
            if image_path:
                print(f"[BlueskyAgent] Auto-posting with image: {image_path}")
                post_result = post_to_bluesky(full_text, image_path=image_path, alt_text=theme)
            else:
                print("[BlueskyAgent] Auto-posting text only")
                post_result = post_to_bluesky(full_text)
        
        return {
            "success": True,
            "text": text,
            "hashtags": hashtags,
            "subliminal": subliminal,
            "image_path": image_path,
            "posted": post_result.get("success") if post_result else False,
            "post_uri": post_result.get("post_uri") if post_result else None,
            "beat_data": beat_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "text": None
        }
