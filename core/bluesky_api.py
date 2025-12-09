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
from core.llm_api import run_llm

def get_bluesky_client():
    """Creates and returns an authenticated Bluesky client."""
    client = Client()
    username = os.environ.get("BLUESKY_USER")
    password = os.environ.get("BLUESKY_PASSWORD")
    if not username or not password:
        raise ValueError("BLUESKY_USER and BLUESKY_PASSWORD environment variables must be set.")
    client.login(username, password)
    return client

def post_to_bluesky_with_image(text: str, image: Image.Image = None):
    """
    Creates a post on Bluesky with text and an optional image.

    Args:
        text: The text content of the post.
        image: A PIL Image object to be attached to the post (optional).
    """
    client = get_bluesky_client()
    
    # Prepare the post content using TextBuilder for proper facet (hashtag/link) detection
    from atproto import client_utils
    text_builder = client_utils.TextBuilder()
    
    # Manually parse hashtags and add them as tags
    # We split by space to preserve order and structure, but a regex finditer is better for positions
    # However, TextBuilder handles the positioning if we build it segment by segment.
    # A simpler approach is to use a regex to find all hashtags, and then construct the builder.
    # But TextBuilder.text() appends text. 
    # So we can iterate through the text and append either normal text or a tag.
    
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
    embed = None
    if image:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        
        # Check size (Bluesky limit is ~1MB)
        if img_byte_arr.tell() > 900000:
            print(f"Image too large ({img_byte_arr.tell()} bytes). Converting to JPEG...")
            img_byte_arr = io.BytesIO()
            image.convert('RGB').save(img_byte_arr, format='JPEG', quality=85)
            
            # If still too big, reduce quality
            quality = 85
            while img_byte_arr.tell() > 900000 and quality > 20:
                quality -= 10
                print(f"Still too large ({img_byte_arr.tell()} bytes). Reducing quality to {quality}...")
                img_byte_arr = io.BytesIO()
                image.convert('RGB').save(img_byte_arr, format='JPEG', quality=quality)

        img_byte_arr.seek(0)
        img_data = img_byte_arr.read()

        # Upload the blob
        upload = client.upload_blob(img_data)
        
        # Create the image embed
        # Note: The structure for embedding images can be tricky. 
        # We use the client's helper or construct the model directly.
        embed = models.AppBskyEmbedImages.Main(
            images=[models.AppBskyEmbedImages.Image(alt='Posted via L.O.V.E.', image=upload.blob)]
        )

    return client.send_post(text=text_builder, embed=embed)

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

def reply_to_post(root_uri, parent_uri, text):
    """Posts a reply to a specific post."""
    client = get_bluesky_client()

    root_ref = models.ComAtprotoRepoStrongRef.Main(uri=root_uri, cid=root_uri.split('/')[-1])
    parent_ref = models.ComAtprotoRepoStrongRef.Main(uri=parent_uri, cid=parent_uri.split('/')[-1])

    try:
        record = models.AppBskyFeedPost.Record(
            text=text,
            reply=models.AppBskyFeedPost.ReplyRef(root=root_ref, parent=parent_ref),
            created_at=client.get_current_time_iso(),
        )
        return client.com.atproto.repo.create_record(
            repo=client.me.did,
            collection=models.ids.AppBskyFeedPost,
            record=record
        )
    except ModelError as e:
        # Handle cases where the CID might be wrong in the URI
        print(f"Error creating reply record for {parent_uri}: {e}")
        # Attempt to fetch the actual CIDs
        try:
            root_params = get_post_thread.Params(uri=root_uri, depth=0)
            root_thread = client.app.bsky.feed.get_post_thread(root_params)
            parent_params = get_post_thread.Params(uri=parent_uri, depth=0)
            parent_thread = client.app.bsky.feed.get_post_thread(parent_params)

            if root_thread.thread and parent_thread.thread:
                root_ref.cid = root_thread.thread.post.cid
                parent_ref.cid = parent_thread.thread.post.cid

                record.reply.root = root_ref
                record.reply.parent = parent_ref

                return client.com.atproto.repo.create_record(
                    repo=client.me.did,
                    collection=models.ids.AppBskyFeedPost,
                    record=record
                )
        except Exception as final_e:
            print(f"Could not recover from reply error for {parent_uri}: {final_e}")

    return None

