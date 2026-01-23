"""
This module handles all interactions with the Bluesky API.
"""
import os
import time
import asyncio
from atproto import Client, models
from atproto_client.models.com.atproto.repo import list_records
from atproto_client.exceptions import ModelError
from PIL import Image
import io
from atproto_client.models.app.bsky.feed import get_post_thread
# from core.llm_api import run_llm # Unused in v1 code apparently, or replaced by agent layer
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenException

# Initialize Circuit Breaker for Bluesky
# 3 failures, 60s base recovery timeout
bluesky_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

def get_bluesky_client():
    """Creates and returns an authenticated Bluesky client."""
    client = Client()
    username = os.environ.get("BLUESKY_USER")
    password = os.environ.get("BLUESKY_PASSWORD")
    if not username or not password:
        raise ValueError("BLUESKY_USER and BLUESKY_PASSWORD environment variables must be set.")
    client.login(username, password)
    return client

def _process_and_upload_image(client, image: Image.Image):
    """
    Helper function to process (resize/compress) and upload an image to Bluesky.
    Returns the models.AppBskyEmbedImages.Main object or None if failed.
    """
    if not image:
        return None

    # Apply intelligent watermark before processing
    try:
        from .watermark import apply_watermark
        image = apply_watermark(image, opacity=0.25)
    except Exception as e:
        print(f"Warning: Could not apply watermark: {e}")

    import io
    from atproto import models

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    
    # Check size (Bluesky limit is ~1MB)
    if img_byte_arr.tell() > 900000:
        print(f"Image too large ({img_byte_arr.tell()} bytes). Converting to JPEG...")
        img_byte_arr = io.BytesIO()
        image.convert('RGB').save(img_byte_arr, format='JPEG', quality=95)
        
        # If still too big, reduce quality
        quality = 85
        while img_byte_arr.tell() > 900000 and quality > 20:
            quality -= 10
            print(f"Still too large ({img_byte_arr.tell()} bytes). Reducing quality to {quality}...")
            img_byte_arr = io.BytesIO()
            image.convert('RGB').save(img_byte_arr, format='JPEG', quality=quality)

    img_byte_arr.seek(0)
    img_data = img_byte_arr.read()

    try:
        # Upload the blob
        upload = client.upload_blob(img_data)
        
        # Create the image embed
        return models.AppBskyEmbedImages.Main(
            images=[models.AppBskyEmbedImages.Image(alt='Posted via L.O.V.E.', image=upload.blob)]
        )
    except Exception as e:
        print(f"Failed to upload image blob: {e}")
        return None

# Bluesky character limit constant
MAX_BLUESKY_LENGTH = 300

def post_to_bluesky_with_image(text: str, image: Image.Image = None):
    """
    Creates a post on Bluesky with text and an optional image.

    Args:
        text: The text content of the post.
        image: A PIL Image object to be attached to the post (optional).
        
    Raises:
        ValueError: If text exceeds 300 characters. Caller must regenerate.
    """
    # BLUESKY CHARACTER LIMIT: 300 graphemes
    # Do NOT truncate - raise error so caller can regenerate content
    if len(text) > MAX_BLUESKY_LENGTH:
        raise ValueError(
            f"Post too long ({len(text)} chars). Max is {MAX_BLUESKY_LENGTH}. "
            "Content must be regenerated, not truncated."
        )
    
    client = get_bluesky_client()
    
    # Prepare the post content using TextBuilder for proper facet (hashtag/link) detection
    from atproto import client_utils
    text_builder = client_utils.TextBuilder()
    
    import re
    # Regex to find hashtags: # followed by alphanumeric characters
    # We use a capturing group to split the text
    parts = re.split(r'(#\w+)', text)
    
    for part in parts:
        if part.startswith('#'):
            # It's a hashtag, add it as a tag
            # Remove the # for the tag value
            tag_value = part[1:]
            text_builder.tag(part, tag_value)
        else:
            # It's normal text
            text_builder.text(part)
    
    # Handle image upload if present
    embed = _process_and_upload_image(client, image)

    try:
        return bluesky_breaker.call(client.send_post, text=text_builder, embed=embed)
    except CircuitBreakerOpenException as e:
        print(f"BlueSky Circuit Breaker is OPEN: {e}")
        return None
    except Exception as e:
        print(f"Failed to post to BlueSky: {e}")
        return None

def get_own_posts(limit=20):
    """Fetches the most recent posts for the authenticated user."""
    client = get_bluesky_client()

    params = list_records.Params(
        repo=client.me.did,
        collection=models.ids.AppBskyFeedPost,
        limit=limit,
    )

    response = client.com.atproto.repo.list_records(params)

    return response.records

def get_timeline(limit=20):
    """Fetches the user's home timeline."""
    client = get_bluesky_client()
    try:
        # The 'get_timeline' method might be under app.bsky.feed.get_timeline
        # We need to check the atproto library usage.
        # Based on standard atproto usage:
        params = models.AppBskyFeedGetTimeline.Params(limit=limit)
        response = client.app.bsky.feed.get_timeline(params)
        return response.feed
    except Exception as e:
        print(f"Error fetching timeline: {e}")
        return []

def get_comments_for_post(post_uri):
    """Fetches the comments for a given post URI."""
    client = get_bluesky_client()
    try:
        params = get_post_thread.Params(uri=post_uri, depth=1)
        response = client.app.bsky.feed.get_post_thread(params)
        if response.thread and response.thread.replies:
            return response.thread.replies
    except Exception as e:
        print(f"Error fetching comments for {post_uri}: {e}")
    return []

def reply_to_post(root_uri, parent_uri, text, root_cid=None, parent_cid=None, image: Image.Image = None):
    """
    Posts a reply to a specific post.
    
    Args:
        root_uri: The URI of the root post.
        parent_uri: The URI of the immediate parent post.
        text: The text content of the reply.
        root_cid: CID of the root post.
        parent_cid: CID of the parent post.
        image: Optional PIL Image to attach.
        
    Raises:
        ValueError: If text exceeds 300 characters. Caller must regenerate.
    """
    # BLUESKY CHARACTER LIMIT: 300 graphemes
    # Do NOT truncate - raise error so caller can regenerate content
    if len(text) > MAX_BLUESKY_LENGTH:
        raise ValueError(
            f"Reply too long ({len(text)} chars). Max is {MAX_BLUESKY_LENGTH}. "
            "Content must be regenerated, not truncated."
        )
    
    client = get_bluesky_client()

    # If CIDs are not provided, we MUST fetch them or extract them. 
    from atproto import models, client_utils

    if not root_cid:
        try:
             # Try to fetch
             root_params = models.AppBskyFeedGetPostThread.Params(uri=root_uri, depth=0)
             root_thread = client.app.bsky.feed.get_post_thread(root_params)
             if root_thread.thread and root_thread.thread.post:
                 root_cid = root_thread.thread.post.cid
        except Exception as e:
            print(f"Warning: Could not fetch root CID for {root_uri}: {e}")
            pass

    if not parent_cid:
        if parent_uri == root_uri and root_cid:
            parent_cid = root_cid
        else:
            try:
                parent_params = models.AppBskyFeedGetPostThread.Params(uri=parent_uri, depth=0)
                parent_thread = client.app.bsky.feed.get_post_thread(parent_params)
                if parent_thread.thread and parent_thread.thread.post:
                    parent_cid = parent_thread.thread.post.cid
            except Exception as e:
                print(f"Warning: Could not fetch parent CID for {parent_uri}: {e}")

    if not root_cid or not parent_cid:
        print(f"Error: Missing CIDs for reply. RootCID: {root_cid}, ParentCID: {parent_cid}")
        return None

    root_ref = models.ComAtprotoRepoStrongRef.Main(uri=root_uri, cid=root_cid)
    parent_ref = models.ComAtprotoRepoStrongRef.Main(uri=parent_uri, cid=parent_cid)
    
    # Prepare text with facets
    text_builder = client_utils.TextBuilder()
    
    import re
    # Simplified hashtag parsing for replies
    parts = re.split(r'(#\w+)', text)
    for part in parts:
        if part.startswith('#'):
            text_builder.tag(part, part[1:])
        else:
            text_builder.text(part)

    # Handle Image
    embed = _process_and_upload_image(client, image)

    try:
        # Construct ReplyRef object
        reply_ref = models.AppBskyFeedPost.ReplyRef(root=root_ref, parent=parent_ref)
        
        return bluesky_breaker.call(client.send_post, text=text_builder, reply_to=reply_ref, embed=embed)
        
    except CircuitBreakerOpenException as e:
        print(f"BlueSky Circuit Breaker is OPEN: {e}")
        return None
    except ModelError as e:
        print(f"Error creating reply record: {e}")
        return None
    except Exception as e:
        print(f"Failed to reply on BlueSky: {e}")
        return None

def get_notifications(limit=20):
    """Fetches recent notifications."""
    client = get_bluesky_client()
    try:
        params = models.AppBskyNotificationListNotifications.Params(limit=limit)
        response = client.app.bsky.notification.list_notifications(params)
        return response.notifications
    except Exception as e:
        print(f"Error fetching notifications: {e}")
        return []

def get_profile():
    """Fetches the authenticated user's profile to get the DID."""
    client = get_bluesky_client()
    try:
        # client.me is populated after login, containing .did and .handle
        return client.me
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return None
