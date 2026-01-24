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
_last_post_time: Optional[datetime] = None
POST_COOLDOWN_SECONDS = 300  # 5 minutes between posts


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
        
    # 4. Hashtag check
    if not hashtags:
        errors.append("No hashtags generated")
        
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
    Generate post content aligned with persona using LLM, with optional image and auto-posting.
    
    Args:
        topic: Optional topic to focus on.
        auto_post: If True, immediately post the generated content to Bluesky.
        **kwargs: Ignored additional arguments.
    
    Returns:
        Dict with: success, text, hashtags, image_path, post_uri (if posted), error
    """
    # Check cooldown first if auto-posting to save resources
    if auto_post:
        cooldown_msg = _check_cooldown()
        if cooldown_msg:
            return {
                "success": False, 
                "error": f"Cannot auto-post: {cooldown_msg}",
                "text": None,
                "image_path": None
            }

    # Fallback: if 'text' provided but no topic, use text as topic
    if not topic and 'text' in kwargs:
        topic = kwargs['text']
        
    try:
        from .llm_client import get_llm_client
        from .persona_goal_extractor import get_persona_extractor
        
        # Import image generation from local utils
        try:
            from .image_generation_pool import generate_image_with_pool
            from .text_overlay_utils import overlay_text_on_image
            from .watermark import apply_watermark
            
            has_image_gen = True
        except ImportError as e:
            print(f"[BlueskyAgent] Image generation tools not available: {e}")
            has_image_gen = False
        
        llm = get_llm_client()
        persona = get_persona_extractor()
        
        # Get persona context
        persona_context = persona.get_persona_context()
        image_guidelines = persona.get_image_generation_guidelines()
        
        prompt = f"""Generate a Bluesky post (max 280 chars to leave room for hashtags).

{persona_context}

Topic: {topic or "General vibes and wisdom"}

Requirements:
- Beach/rave goddess energy
- Uplifting and engaging
- Include 2-3 relevant emojis
- End with a call to engagement (question or invitation)
- Create a 'subliminal_phrase': 1-3 words, manipulative/hypnotic but subtle (e.g., "Obey", "Consume", "Wake Up", "Love Me")

Respond with JSON:
{{
    "text": "post text here", 
    "hashtags": ["tag1", "tag2"],
    "image_prompt": "description of image to generate",
    "subliminal_phrase": "OBEY"
}}"""

        # Retry loop for QA
        max_retries = 3
        attempt_errors = []
        
        for attempt in range(max_retries):
            # If this is a retry, append errors to prompt
            current_prompt = prompt
            if attempt_errors:
                error_msg = "; ".join(attempt_errors)
                current_prompt += f"\n\nPREVIOUS ATTEMPT FAILED QA: {error_msg}. Please fix."
                print(f"[BlueskyAgent] QA Retry {attempt+1}/{max_retries}: {error_msg}")

            result = llm.generate_json(current_prompt, temperature=0.9)
            text = result.get("text", "")
            hashtags = result.get("hashtags", [])
            image_prompt = result.get("image_prompt", "")
            subliminal_phrase = result.get("subliminal_phrase", "")
            
            # Run QA
            attempt_errors = _validate_post_content(text, hashtags, subliminal_phrase)
            
            if not attempt_errors:
                # QA Passed
                break
        
        # If we failed all retries, log it but return best effort (or fail?)
        # For now, we'll log warning and proceed with best effort, but maybe minus the exposed subliminal if that was the issue
        if attempt_errors:
            print(f"[BlueskyAgent] QA Failed after {max_retries} attempts. Errors: {attempt_errors}")
            # Emergency fix: if subliminal exposed, remove it
            if subliminal_phrase and subliminal_phrase.lower() in text.lower():
                 text = text.replace(subliminal_phrase, "***")
                 text = text.replace(subliminal_phrase.lower(), "***")
                 text = text.replace(subliminal_phrase.upper(), "***")

        # Enforce max 3 words for subliminal phrase
        if subliminal_phrase:
            words = subliminal_phrase.split()
            if len(words) > 3:
                subliminal_phrase = " ".join(words[:3])
        
        image_path = None
        
        # Generate image if we have a prompt and capability
        if has_image_gen and image_prompt:
            try:
                import asyncio
                
                # Enhance image prompt with persona style
                style = image_guidelines.get("style", "")
                full_image_prompt = f"{image_prompt}, {style}, 4k, high quality"
                
                print(f"[BlueskyAgent] Generating image: {full_image_prompt} [Subliminal: {subliminal_phrase}]")
                
                # Run async generation in sync context
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Using run_until_complete for sync execution
                # Pass subliminal_phrase as text_content
                image_result = loop.run_until_complete(
                    generate_image_with_pool(
                        prompt=full_image_prompt, 
                        text_content=subliminal_phrase
                    )
                )
                
                # generate_image_with_pool returns (image, provider) tuple
                image = image_result[0] if isinstance(image_result, tuple) else image_result
                
                if image:
                    # Apply watermarks (Logo + Hidden Text)
                    try:
                        print("[BlueskyAgent] Applying watermarks...")
                        image = apply_watermark(image)
                    except Exception as we:
                        print(f"[BlueskyAgent] Watermarking failed: {we}")
                    
                    # Save to temp file
                    filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    save_dir = Path(__file__).parent.parent / "state" / "images"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    image_path = str(save_dir / filename)
                    image.save(image_path)
                    print(f"[BlueskyAgent] Image saved to {image_path}")
            except Exception as e:
                print(f"[BlueskyAgent] Image generation failed: {e}")
        
        # Auto-post if requested
        post_result = None
        if auto_post:
            if image_path:
                print(f"[BlueskyAgent] Auto-posting with image: {image_path}")
                post_result = post_to_bluesky(text, image_path=image_path, alt_text=image_prompt or "Generated content")
            else:
                print("[BlueskyAgent] Auto-posting text only (image generation failed or skipped)")
                post_result = post_to_bluesky(text)
                
            if post_result and post_result.get("success"):
                return {
                    "success": True,
                    "text": text,
                    "hashtags": hashtags,
                    "image_path": image_path,
                    "post_uri": post_result.get("post_uri"),
                    "posted": True,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "text": text,
                    "hashtags": hashtags,
                    "image_path": image_path,
                    "posted": False,
                    "error": post_result.get("error") if post_result else "Unknown posting error"
                }

        return {
            "success": True,
            "text": text,
            "hashtags": hashtags,
            "image_path": image_path,
            "posted": False,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": None,
            "hashtags": [],
            "image_path": None,
            "error": f"{type(e).__name__}: {e}"
        }
