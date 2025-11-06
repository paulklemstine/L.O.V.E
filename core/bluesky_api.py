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

def post_to_bluesky_with_image(text: str, image: Image.Image):
    """
    Creates a post on Bluesky with text and an image.

    Args:
        text: The text content of the post.
        image: A PIL Image object to be attached to the post.
    """
    client = get_bluesky_client()
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    upload = client.com.atproto.repo.upload_blob(img_byte_arr.read())
    embed = models.AppBskyEmbedImages.Main(images=[models.AppBskyEmbedImages.Image(alt='', image=upload.blob)])

    return client.com.atproto.repo.create_record(
        repo=client.me.did,
        collection=models.ids.AppBskyFeedPost,
        record=models.AppBskyFeedPost.Main(text=text, embed=embed, created_at=client.get_current_time_iso())
    )

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
        record = models.AppBskyFeedPost.Main(
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

