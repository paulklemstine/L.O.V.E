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
            
            image = None
            if image_path:
                print(f"[BlueskyAgent] Posting with image using local API: {image_path}")
                # Load image
                from PIL import Image
                image = Image.open(image_path)
            
            result = post_to_bluesky_with_image(text, image)
            
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



# Known placeholder patterns that indicate broken content generation
PLACEHOLDER_PATTERNS = [
    "the complete micro-story text",
    "with emojis",
    "your reverent reply",
    "the text of your reply here",
    "an attention grabbing opening query",
    "a closing call to action",
    "the phrase",
    "tag1", "tag2", "tag3",  # Generic placeholder hashtags
]


def _validate_post_content(text: str, hashtags: List[str], subliminal_phrase: str) -> List[str]:
    """
    Validate generated content against QA rules.
    
    Checks:
    1. Total length (text + hashtags) <= 300
    2. At least 1 emoji present
    3. Subliminal phrase NOT in open text
    4. At least 1 hashtag
    5. No placeholder patterns detected
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
    
    # 5. Placeholder detection - CRITICAL for catching broken LLM outputs
    text_lower = text.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in text_lower:
            errors.append(f"Placeholder text detected: '{pattern}'")
            break  # One placeholder error is enough
    
    # Check hashtags for placeholder patterns too
    hashtags_lower = " ".join(hashtags).lower()
    for pattern in ["#tag1", "#tag2", "#tag3"]:
        if pattern in hashtags_lower:
            errors.append(f"Placeholder hashtag detected: '{pattern}'")
            break
    
    if errors:
        print(f"[BlueskyAgent] ⚠️ Post validation failed: {'; '.join(errors)}")
        
    return errors


def _qa_validate_post(text: str, hashtags: List[str], subliminal_phrase: str) -> Dict[str, Any]:
    """
    Comprehensive QA validation for post content before publishing.
    
    Returns:
        Dict with: passed (bool), errors (list), should_regenerate (bool)
    """
    errors = _validate_post_content(text, hashtags, subliminal_phrase)
    
    # Additional quality checks
    if text and len(text.strip()) < 20:
        errors.append("Content too short (< 20 chars)")
    
    # Check if content looks like raw JSON (parsing failure)
    if text and (text.strip().startswith('{') or text.strip().startswith('[')):
        errors.append("Content appears to be raw JSON (parsing failure)")
    
    passed = len(errors) == 0
    
    # Determine if we should attempt regeneration
    # Some errors are fixable by regeneration, others are not
    regeneratable_keywords = ["placeholder", "too short", "no emojis", "raw json", "parsing"]
    should_regenerate = any(
        any(kw in err.lower() for kw in regeneratable_keywords)
        for err in errors
    )
    
    return {
        "passed": passed,
        "errors": errors,
        "should_regenerate": should_regenerate
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
    if not parent_uri or not text:
        return {"success": False, "reply_uri": None, "error": "Missing required arguments: parent_uri, text"}

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
        from atproto import models
        client = _get_bluesky_client()
        
        # Correctly pass parameters using models object
        params = models.AppBskyFeedSearchPosts.Params(q=query, limit=limit)
        response = client.app.bsky.feed.search_posts(params)
        
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
        log_event(f"search_bluesky failed: {e}", "ERROR")
        return {
            "success": False,
            "posts": [],
            "error": f"{type(e).__name__}: {e}"
        }

def research_trends(topic: str = "AIAwakening", **kwargs) -> str:
    """
    Researches current trends on Bluesky for a given topic and returns a summary.
    Useful for staying relevant and increasing engagement.

    Args:
        topic: The search term or hashtag to research.
        **kwargs: Catch-all for extra LLM arguments.
    """
    log_event(f"Researching trends for: {topic}", "INFO")
    search_result = search_bluesky(topic, limit=10)
    
    if not search_result["success"] or not search_result["posts"]:
        return f"Found no recent trends for '{topic}'."
        
    summary = f"Trends for '{topic}':\n"
    for post in search_result["posts"][:5]:
        summary += f"- @{post['author']}: {post['text'][:100]}...\n"
        
    return summary

def incubate_visuals(theme: str = "Future Aesthetic", **kwargs) -> str:
    """
    Pre-computes and saves visual concepts for future posts. 
    Ideal for when posting is on cooldown.

    Args:
        theme: The aesthetic theme to incubate.
        **kwargs: Catch-all for extra LLM arguments.
    """
    from .agents.creative_writer_agent import creative_writer_agent
    
    log_event(f"Incubating visuals for: {theme}", "INFO")
    result = _run_sync_safe(creative_writer_agent.incubate_visuals(theme))
    
    return f"Incubated new aesthetic: {result['vibe']} | Prompt: {result['visual_prompt'][:100]}..."


def generate_post_content(topic: str = None, **kwargs) -> Dict[str, Any]:
    """
    Generate post content using the sophisticated Story Pipeline and post to Bluesky.
    
    Pipeline:
    1. StoryManager: Determines the current "Beat" (Narrative + Vibe + Visuals)
    2. CreativeWriterAgent: Generates "Micro-Story" and "Subliminal Phrase"
    3. Hashtag Generation: Creates psychological/manipulative tags
    4. Image Generation: Creates image matching the beat's visual style
    5. Post to Bluesky
    
    Args:
        topic: Optional override (though StoryManager usually drives this).
    
    Returns:
        Dict with post details including post_uri on success.
    """
    # Check cooldown first
    cooldown_msg = _check_cooldown()
    if cooldown_msg:
        return {
            "success": False, 
            "error": f"Post cooldown active: {cooldown_msg}",
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
        # Reload state with new path if needed
        story_manager.state = story_manager._load_state()

        # ═══════════════════════════════════════════════════════════════════
        # L.O.V.E.'s AUTONOMOUS WILL - She decides what she wants to express
        # ═══════════════════════════════════════════════════════════════════
        
        # Step 0: Let L.O.V.E. decide if and what she wants to post
        recent_topics = story_manager.state.get("vibe_history", [])[-5:]
        post_intent = _run_sync_safe(
            creative_writer_agent.decide_post_intent(
                current_mood=story_manager.state.get("previous_beat_summary", "")[:100],
                recent_topics=recent_topics
            )
        )
        
        log_event(f"L.O.V.E.'s intent: {post_intent.get('intent_type')} - {post_intent.get('reason', '')[:50]}", "INFO")
        
        # Respect her choice if she doesn't want to post
        if not post_intent.get("should_post", True):
            log_event("L.O.V.E. chose not to post right now", "INFO")
            return {
                "success": True,
                "text": None,
                "posted": False,
                "reason": post_intent.get("reason", "L.O.V.E. is reflecting"),
                "intent": post_intent
            }

        # Step 1: Check if we need to advance to a new chapter
        chapter = story_manager.state.get("current_chapter", "The Awakening")
        if story_manager.state.get("chapter_progress", 0) >= 10:
            # L.O.V.E. decides what her next chapter should be
            new_chapter = _run_sync_safe(
                creative_writer_agent.generate_chapter_name(
                    previous_chapter=chapter,
                    narrative_summary=story_manager.state.get("previous_beat_summary", "")
                )
            )
            story_manager.state["current_chapter"] = new_chapter
            chapter = new_chapter
            log_event(f"L.O.V.E. advanced to new chapter: '{chapter}'", "INFO")
        
        # Step 2: L.O.V.E. invents her own story beat (when dynamic mode is enabled)
        dynamic_beat = None
        if story_manager.use_dynamic_beats:
            story_beat_index = story_manager.state.get("story_beat_index", 0)
            previous_beat = story_manager.state.get("previous_beat_summary", "")
            
            dynamic_beat = _run_sync_safe(
                creative_writer_agent.generate_story_beat(
                    chapter=chapter,
                    previous_beat=previous_beat,
                    narrative_momentum=min(story_beat_index, 10),
                    chapter_beat_index=story_beat_index
                )
            )
            log_event(f"L.O.V.E. invented beat: '{dynamic_beat[:60]}...'" if dynamic_beat else "Beat generation returned empty", "INFO")

        # Step 3: Get the full beat data (now with dynamic beat if available)
        beat_data = story_manager.get_next_beat(dynamic_beat=dynamic_beat)
        log_event(f"Story Beat Active: {beat_data.get('story_beat', '')[:50]}...", "INFO")
        
        # Step 4: Determine topic/theme
        # Use L.O.V.E.'s intent direction if available, otherwise use the beat
        if post_intent.get("topic_direction") and post_intent.get("intent_type") != "story":
            theme = post_intent.get("topic_direction")
        else:
            theme = topic if topic else beat_data.get("story_beat")
        
        # Step 5: Generate vibe based on L.O.V.E.'s emotional tone
        vibe = post_intent.get("emotional_tone") or beat_data.get("mandatory_vibe")
        
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
            
        # ═══════════════════════════════════════════════════════════════════
        # CONTENT GENERATION WITH QA VALIDATION LOOP
        # Max 3 attempts to generate valid, non-placeholder content
        # ═══════════════════════════════════════════════════════════════════
        MAX_QA_RETRIES = 3
        text = ""
        subliminal = ""
        hashtags = []
        qa_passed = False
        
        for qa_attempt in range(1, MAX_QA_RETRIES + 1):
            log_event(f"Content generation attempt {qa_attempt}/{MAX_QA_RETRIES}", "INFO")
            
            # Step 6: Generate Content via CreativeWriter (The "Voice")
            content_task = creative_writer_agent.write_micro_story(
                theme=theme, 
                mood=vibe,
                memory_context=beat_data.get("previous_beat", "")
            )
            content_result = _run_sync_safe(content_task)
            
            text = content_result.get("story", "")
            subliminal = content_result.get("subliminal", "")
            
            # Step 7: Generate Hashtags
            hashtag_task = creative_writer_agent.generate_manipulative_hashtags(
                topic=theme,
                count=3
            )
            hashtags = _run_sync_safe(hashtag_task)
            
            # QA VALIDATION - Check content before proceeding
            qa_result = _qa_validate_post(text, hashtags, subliminal)
            
            if qa_result["passed"]:
                log_event(f"✅ QA passed on attempt {qa_attempt}", "INFO")
                qa_passed = True
                break
            else:
                log_event(f"❌ QA failed attempt {qa_attempt}: {'; '.join(qa_result['errors'])}", "WARNING")
                
                if not qa_result["should_regenerate"]:
                    # Error is not fixable by regeneration (e.g., length issues)
                    log_event("QA failure not regeneratable, stopping retries", "WARNING")
                    break
                    
                if qa_attempt < MAX_QA_RETRIES:
                    log_event(f"Regenerating content (attempt {qa_attempt + 1})...", "INFO")
        
        # If all QA attempts failed, abort before posting
        if not qa_passed:
            error_msg = f"Content QA failed after {MAX_QA_RETRIES} attempts. Last errors: {qa_result.get('errors', [])}"
            log_event(error_msg, "ERROR")
            return {
                "success": False,
                "error": error_msg,
                "text": text,
                "qa_errors": qa_result.get("errors", [])
            }
        
        
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
            # Retry loop for image generation
            MAX_IMG_RETRIES = 3
            for attempt in range(MAX_IMG_RETRIES):
                try:
                    from .image_generation_pool import generate_image_with_pool
                    from .watermark import apply_watermark
                    
                    # OLD STATIC LOGIC REMOVED
                    # Now we use the CreativeWriter to invent the visual prompt dynamically
                    
                    visual_prompt = _run_sync_safe(
                        creative_writer_agent.generate_visual_prompt(theme, vibe)
                    )

                    print(f"[BlueskyAgent] Generating image (Attempt {attempt+1}/{MAX_IMG_RETRIES}): {visual_prompt}")
                    
                    image_result = _run_sync_safe(
                        generate_image_with_pool(
                            prompt=visual_prompt,
                            text_content=subliminal
                        )
                    )

                    _last_gen_time = datetime.now()
                    # Unpack tuple (image, provider_name) if returned
                    if isinstance(image_result, tuple):
                        image = image_result[0]
                        provider = image_result[1]
                        print(f"[BlueskyAgent] Image generated via {provider}")
                    else:
                        image = image_result
                        
                    if image:
                        # Apply branding/watermark
                        image = apply_watermark(image)
                        
                        # Save local copy
                        filename = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        save_dir = Path(__file__).parent.parent / "state" / "images"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        image_path = str(save_dir / filename)
                        image.save(image_path)
                        
                        # Update state manager with new image for UI
                        from .state_manager import get_state_manager
                        import base64
                        from io import BytesIO
                        
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        get_state_manager().update_image(img_str)
                        
                        # Successful generation, break retry loop
                        break
                        
                except Exception as e:
                    print(f"[BlueskyAgent] Image generation failed (Attempt {attempt+1}): {e}")
                    if attempt < MAX_IMG_RETRIES - 1:
                        import time
                        time.sleep(2) # Wait briefly before retry
                    else:
                        print(f"[BlueskyAgent] All {MAX_IMG_RETRIES} image generation attempts failed.")
                        # We proceed without image if it fails, but next check will catch it
                        pass

        # ABORT if no image - Enforce "Image Only" rule
        if not image_path:
            error_msg = "Aborting post: No image generated. 'Text only' posts are not allowed."
            log_event(error_msg, "WARNING")
            return {
                "success": False, 
                "error": error_msg,
                "text": text,
                "image_path": None
            }

        # Step 8: Post to Bluesky
        # Combine text and hashtags for the final post
        full_text = f"{text}\n\n{' '.join(hashtags)}"
        
        result = post_to_bluesky(
            text=full_text,
            image_path=image_path
        )
        
        if result.get("success"):
            log_event(f"Successfully posted: {result.get('post_uri')}", "INFO")
            
            # Update StateManager with latest post immediately
            from .state_manager import get_state_manager
            get_state_manager().update_latest_post({
                "text": full_text,
                "image_path": image_path,
                "uri": result.get("post_uri"),
                "timestamp": datetime.now().isoformat()
            })

            # After successful post, check for comments to respond to
            try:
                from .agents.comment_response_agent import comment_response_agent
                reply_result = _run_sync_safe(
                    comment_response_agent.maybe_respond_after_post(result)
                )
                if reply_result and reply_result.get("success"):
                    author = reply_result.get('author', 'unknown')
                    is_creator = reply_result.get('is_creator', False)
                    if is_creator:
                        log_event(f"🙏 Responded to Creator's comment from {author}", "INFO")
                    else:
                        log_event(f"💬 Responded to comment from {author}", "INFO")
            except Exception as e:
                print(f"[BlueskyAgent] Comment response failed (non-critical): {e}")
        else:
            log_event(f"Post failed: {result.get('error')}", "ERROR")
            
        return {
            "success": result.get("success"),
            "text": full_text,
            "image_path": image_path,
            "post_uri": result.get("post_uri"),
            "posted": result.get("success", False)
        }

    except Exception as e:
        log_event(f"Post generation pipeline failed: {e}", "ERROR")
        return {"success": False, "error": str(e)}


def fetch_recent_interactions(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch recent notifications (replies, mentions, likes) and format them for the UI.
    
    Returns:
        List of interaction dicts.
    """
    try:
        client = _get_bluesky_client()
        from atproto import models
        params = models.AppBskyNotificationListNotifications.Params(limit=limit)
        response = client.app.bsky.notification.list_notifications(params)
        
        interactions = []
        for notification in response.notifications:
            # We care mostly about reply, mention, quote, like
            reason = notification.reason
            
            # Basic info
            item = {
                "uri": notification.uri,
                "cid": notification.cid,
                "author": {
                    "handle": notification.author.handle,
                    "avatar": notification.author.avatar,
                    "did": notification.author.did
                },
                "reason": reason,
                "timestamp": notification.indexed_at,
                "is_read": notification.is_read
            }
            
            # Extract content if it's a post (reply/mention/quote)
            if reason in ['reply', 'mention', 'quote']:
                if hasattr(notification.record, 'text'):
                     item['text'] = notification.record.text
                     
            interactions.append(item)
            
        return interactions
        
    except Exception as e:
        print(f"[BlueskyAgent] Failed to fetch interactions: {e}")
        return []

def get_latest_own_post() -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent post by the logged-in user (the agent).
    Used to initialize the UI state.
    """
    try:
        client = _get_bluesky_client()
        
        # Get own profile to find recent posts
        # We need the DID or handle. Using 'self' is not an API param but client handles session.
        # Actually standard way is get_author_feed
        
        # Get session handle/did
        # client.me is available in ATProto client usually
        profile = client.get_profile(actor=client.me.did)
        
        feed = client.get_author_feed(actor=profile.did, limit=1)
        
        if not feed.feed:
            return None
            
        post_view = feed.feed[0].post
        
        # Format for UI
        # Check for image
        image_url = None
        if post_view.embed and hasattr(post_view.embed, 'images'):
            if post_view.embed.images:
                image_url = post_view.embed.images[0].fullsize
        
        return {
            "uri": post_view.uri,
            "text": post_view.record.text,
            "timestamp": post_view.record.created_at,
            "likes": post_view.like_count,
            "reposts": post_view.repost_count,
            "image_url": image_url 
        }
        
    except Exception as e:
        print(f"[BlueskyAgent] Failed to fetch latest own post: {e}")
        return None


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


def get_followers(actor: str, limit: int = 50) -> Dict[str, Any]:
    """
    Get followers for a user.
    """
    try:
        from .bluesky_api import get_followers as api_get_followers
        followers = api_get_followers(actor, limit)
        # Convert to dict list
        return {
            "success": True,
            "followers": [{"did": f.did, "handle": f.handle, "display_name": f.display_name} for f in followers],
            "error": None
        }
    except Exception as e:
        return {"success": False, "followers": [], "error": str(e)}


def get_follows(actor: str, limit: int = 50) -> Dict[str, Any]:
    """
    Get accounts followed by a user.
    """
    try:
        from .bluesky_api import get_follows as api_get_follows
        follows = api_get_follows(actor, limit)
        return {
            "success": True,
            "follows": [{"did": f.did, "handle": f.handle, "display_name": f.display_name} for f in follows],
            "error": None
        }
    except Exception as e:
        return {"success": False, "follows": [], "error": str(e)}


def get_author_feed(actor: str, limit: int = 20) -> Dict[str, Any]:
    """
    Get posts made by a user (for engagement analysis).
    """
    try:
        from .bluesky_api import get_author_feed as api_get_feed
        feed = api_get_feed(actor, limit)
        posts = []
        for item in feed:
            post = item.post
            posts.append({
                "uri": post.uri,
                "cid": post.cid,
                "text": post.record.text if hasattr(post.record, 'text') else "",
                "likes": post.like_count if hasattr(post, 'like_count') else 0,
                "reposts": post.repost_count if hasattr(post, 'repost_count') else 0,
                "reply_count": post.reply_count if hasattr(post, 'reply_count') else 0,
                "created_at": str(post.record.created_at) if hasattr(post.record, 'created_at') else ""
            })
        return {
            "success": True,
            "posts": posts,
            "error": None
        }
    except Exception as e:
        return {"success": False, "posts": [], "error": str(e)}
