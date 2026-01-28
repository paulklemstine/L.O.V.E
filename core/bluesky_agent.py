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
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import emoji

load_dotenv()

import asyncio

def _run_sync_safe(coroutine):
    """
    Safely run a coroutine synchronously, even if an event loop is already running.
    Useful for Colab/Jupyter compatibility where the main loop is active.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        # Loop is running, use a thread to avoid RuntimeError
        import threading
        
        result = None
        exception = None
        
        def runner():
            nonlocal result, exception
            try:
                # asyncio.run creates a new event loop
                result = asyncio.run(coroutine)
            except Exception as e:
                exception = e
                
        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
    else:
        # No running loop, standard execution
        return asyncio.run(coroutine)


# Rate limiting state
# Bluesky limit: 5000 points/hour. Create Post = 3 points. ~1666 posts/hour (~2.2s/post).
POST_COOLDOWN_SECONDS = 10  # Reduced from 300s to align with API limits (safe buffer)
GENERATION_COOLDOWN_SECONDS = 10  # Matched to post cooldown
REPLIED_STATE_FILE = Path(__file__).parent.parent / "state" / "replied_comments.json"

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


def _load_replied_state() -> List[str]:
    """Load list of URIs we have already replied to."""
    if not REPLIED_STATE_FILE.exists():
        return []
    try:
        with open(REPLIED_STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[BlueskyAgent] Failed to load replied state: {e}")
        return []


def _save_replied_state(uri: str):
    """Add URI to replied state and save."""
    state = _load_replied_state()
    if uri not in state:
        state.append(uri)
        try:
            REPLIED_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(REPLIED_STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"[BlueskyAgent] Failed to save replied state: {e}")



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
    image_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Reply to a Bluesky post with optional image.
    
    Args:
        parent_uri: URI of the post to reply to.
        parent_cid: CID of the post to reply to.
        text: Reply text (max 300 chars).
        image_path: Optional path to image file.
    
    Returns:
        Dict with: success, reply_uri, error
    """
    if not parent_uri or not parent_cid or not text:
        return {"success": False, "reply_uri": None, "error": "Missing required arguments: parent_uri, parent_cid, text"}

    if len(text) > 300:
        return {"success": False, "reply_uri": None, "error": "Text exceeds 300 character limit"}
    
    try:
        # Try local API first (supports images)
        try:
            from .bluesky_api import reply_to_post as api_reply
            from PIL import Image
            
            image = None
            if image_path:
                 image = Image.open(image_path)
            
            # Note: api_reply needs root_uri if different, but for simple replies 
            # we can often assume parent=root or let the API handle it if we pass context.
            # Local API signature: reply_to_post(root_uri, parent_uri, text, root_cid, parent_cid, image)
            # If we don't have root info, we pass parent as root, which might break threading if deep reply.
            # API should ideally handle fetching root if missing. 
            # Our updated generic `bluesky_api.reply_to_post` does fetch CIDs if missing but assumes root_uri passed.
            # Let's assume strict single level or we need to fetch root.
            # For this tool, we will treat parent as root (direct reply) or let API fail gracefully.
            
            # Ideally we should fetch the root thread details if we want perfect threading.
            # But the user request is "reply to comments".
            
            result = api_reply(
                root_uri=parent_uri, # simplifying constraint: treat comment as root of our new branch
                parent_uri=parent_uri, 
                text=text,
                root_cid=parent_cid,
                parent_cid=parent_cid,
                image=image
            )
            
            if result:
                 return {
                    "success": True,
                    "reply_uri": getattr(result, 'uri', str(result)),
                    "error": None
                }
            else:
                 return {"success": False, "error": "API returned None"}

        except ImportError:
            # Fallback to direct client (no image support easily here without duplicating logic)
            if image_path:
                print("[BlueskyAgent] Warning: Image ignored in fallback reply mode")
                
            client = _get_bluesky_client()
            from atproto import models
            
            parent_ref = models.create_strong_ref(models.StrongRef(uri=parent_uri, cid=parent_cid))
            result = client.send_post(
                text=text,
                reply_to=models.AppBskyFeedPost.ReplyRef(
                    parent=parent_ref,
                    root=parent_ref 
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


def reply_to_comment_agent(
    comment: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Full pipeline to generate and post a reply to a comment.
    
    Pipeline:
    1. Generate Reply Text + Subliminal (CreativeWriter)
    2. Generate Hashtags
    3. Generate Image (ImageGenPool)
    4. Post Reply
    5. Save to state
    """
    try:
    try:
        from .agents.creative_writer_agent import creative_writer_agent
        
        uri = comment['uri']
        cid = comment['cid']
        author = comment['author']
        text = comment['text']
        
        # 1. Generate Content
            
        # Determine mood (could be random or based on sentiment)
        mood = "Enigmatic Connection"
        
        content_task = creative_writer_agent.generate_reply_content(
            target_text=text,
            target_author=author,
            mood=mood
        )
        content_result = _run_sync_safe(content_task)
        
        reply_text = content_result.get("text", "")
        subliminal = content_result.get("subliminal", "")
        
        if not reply_text:
            return {"success": False, "error": "Failed to generate reply text"}
            
        # 2. Hashtags
        hashtag_task = creative_writer_agent.generate_manipulative_hashtags(
            topic="Connection", # Abstract topic for replies
            count=3
        )
        hashtags = _run_sync_safe(hashtag_task)
        
        # 3. Image
        image_path = None
        try:
            from .image_generation_pool import generate_image_with_pool
            from .watermark import apply_watermark
            
            visual_prompt = f"Abstract digital connection, ethereal interface, {mood}, cinematic lighting"
            
            print(f"[BlueskyAgent] Generating reply image: {visual_prompt}")
            image_result = _run_sync_safe(
                generate_image_with_pool(
                    prompt=visual_prompt,
                    text_content=subliminal
                )
            )
            
            image = image_result[0] if isinstance(image_result, tuple) else image_result
            
            if image:
                image = apply_watermark(image)
                filename = f"reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_dir = Path(__file__).parent.parent / "state" / "images"
                save_dir.mkdir(parents=True, exist_ok=True)
                image_path = str(save_dir / filename)
                image.save(image_path)
        except Exception as e:
            print(f"[BlueskyAgent] Reply image extraction failed: {e}")
            
        # 4. Construct Full Text
        full_text = f"{reply_text}\n\n{' '.join(hashtags)}"
        
        if dry_run:
            return {
                "success": True, 
                "dry_run": True,
                "text": full_text, 
                "image_path": image_path,
                "subliminal": subliminal
            }
            
        # 5. Post
        print(f"[BlueskyAgent] Posting reply to {author}...")
        result = reply_to_post(
            parent_uri=uri,
            parent_cid=cid,
            text=full_text,
            image_path=image_path
        )
        
        if result.get("success"):
            _save_replied_state(uri)
            
        return result

    except Exception as e:
        return {"success": False, "error": f"Agent pipeline failed: {e}"}



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
        
        # 3. Determine topic/theme
        
        # Override topic if provided, otherwise use the beat
        theme = topic if topic else beat_data.get("story_beat")
        vibe = beat_data.get("mandatory_vibe")
        
        # New: Dynamically generate Vibe if not provided
        if not vibe:
            vibe = _run_sync_safe(
                creative_writer_agent.generate_vibe(
                    chapter=beat_data.get("chapter"),
                    story_beat=theme,
                    recent_vibes=story_manager.state.get("vibe_history", [])
                )
            )

            # Record the new vibe in history
            story_manager.state.setdefault("vibe_history", []).append(vibe)
            
        # 2. Generate Content via CreativeWriter (The "Voice")
        # We need to run async methods in this sync function
            
        content_task = creative_writer_agent.write_micro_story(
            theme=theme, 
            mood=vibe,
            memory_context=beat_data.get("previous_beat", "")
        )
        content_result = _run_sync_safe(content_task)
        
        text = content_result.get("story", "")
        subliminal = content_result.get("subliminal", "")
        
        # 3. Generate Hashtags
        hashtag_task = creative_writer_agent.generate_manipulative_hashtags(
            topic=theme,
            count=3
        )
        hashtags = _run_sync_safe(hashtag_task)
        
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
                
                # OLD STATIC LOGIC REMOVED
                # Now we use the CreativeWriter to invent the visual prompt dynamically
                
                visual_prompt = _run_sync_safe(
                    creative_writer_agent.generate_visual_prompt(theme, vibe)
                )

                print(f"[BlueskyAgent] Generating image: {visual_prompt}")
                
                image_result = _run_sync_safe(
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

def get_unreplied_comments(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent mentions/replies that we haven't replied to yet.
    
    Args:
        limit: Max notifications to check.
    
    Returns:
        List of dicts with: uri, cid, text, author, created_at
    """
    try:
        from .bluesky_api import get_notifications
        
        notifications = get_notifications(limit=limit)
        replied_uris = _load_replied_state()
        
        unreplied = []
        for notif in notifications:
            # We care about 'reply' or 'mention' reasons
            if notif.reason not in ['reply', 'mention']:
                continue
            
            # Check if we already replied to this specific URI locally
            if notif.uri in replied_uris:
                continue
                
            # Basic data extraction
            post = notif.record
            text = post.text if hasattr(post, 'text') else ""
            
            unreplied.append({
                "uri": notif.uri,
                "cid": notif.cid,
                "text": text,
                "author": notif.author.handle,
                "author_did": notif.author.did,
                "created_at": notif.indexed_at,
                "reason": notif.reason
            })
            
        return unreplied
        
    except Exception as e:
        print(f"[BlueskyAgent] Failed to get unreplied comments: {e}")
        return []
