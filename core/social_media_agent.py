import asyncio
import random
import time
import os
import traceback

from core.bluesky_api import get_own_posts, get_comments_for_post
from core.social_media_react_engine import SocialMediaReActEngine
from core.logging import log_event
from core.dispatcher import dispatch_structured_payload
from core.interface_handlers import BlueskyAPIHandler


class SocialMediaAgent:
    """
    An autonomous agent that manages L.O.V.E.'s social media presence.
    """
    def __init__(self, loop):
        self.loop = loop
        self.engine = SocialMediaReActEngine(ui_panel_queue=None, loop=loop)
        self.processed_comments = set()
        self.max_retries = 3
        self.handlers = {
            'bluesky': BlueskyAPIHandler()
        }
        # Track failed post generation attempts
        self.last_post_failure_time = 0
        self.post_failure_cooldown = 300  # 5 minutes cooldown after failure

    async def _attempt_action(self, action, *args, **kwargs):
        """Wrapper to retry an action up to max_retries times."""
        for attempt in range(self.max_retries):
            try:
                result = await action(*args, **kwargs)
                return result
            except Exception as e:
                log_event(f"Attempt {attempt + 1}/{self.max_retries} failed for action {action.__name__}. Error: {e}", level='WARNING')
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    log_event(f"Action {action.__name__} failed after {self.max_retries} attempts.", level='ERROR')
                    return None

    async def _post_new_content(self, platform: str):
        """Generates and posts new content to a specified platform."""
        handler = self.handlers.get(platform)
        if not handler:
            log_event(f"No handler found for platform: {platform}", level='WARNING')
            return

        log_event(f"Generating new post for {platform}...", level='INFO')
        post_data = await self.engine.run_post_generation()
        
        # Check if post generation failed
        if post_data is None:
            log_event(f"Post generation failed. Entering cooldown period.", level='WARNING')
            self.last_post_failure_time = time.time()
            return
        
        if post_data.get('text'):
            payload = {
                'action': 'post',
                'platform_identifier': platform,
                'content': post_data.get('text'),
                'image': post_data.get('image')
            }
            result = await dispatch_structured_payload(payload, handler)
            log_event(f"Posted to {platform}: {post_data.get('text')}. Image attached: {post_data.get('image') is not None}. Result: {result}", level='INFO')

    async def post_status_update(self, context: str, platform: str = 'bluesky'):
        """
        Manually triggers a status update post with specific context.
        Bypasses the scheduling logic.
        """
        handler = self.handlers.get(platform)
        if not handler:
            log_event(f"No handler found for platform: {platform}", level='WARNING')
            return

        log_event(f"Generating manual status update for {platform} with context: {context}", level='INFO')
        post_data = await self.engine.run_post_generation(context=context)

        # Check if post generation failed
        if post_data is None:
            log_event(f"Manual status update generation failed.", level='WARNING')
            self.last_post_failure_time = time.time()
            return

        if post_data.get('text'):
            payload = {
                'action': 'post',
                'platform_identifier': platform,
                'content': post_data.get('text'),
                'image': post_data.get('image')
            }
            # We use _attempt_action to ensure resilience even for manual posts
            # Use a lambda or partial to pass arguments correctly to dispatch_structured_payload
            action = lambda: dispatch_structured_payload(payload, handler)
            result = await self._attempt_action(action)
            log_event(f"Manual status update posted to {platform}: {post_data.get('text')}. Result: {result}", level='INFO')

    async def _check_and_reply_to_comments(self, platform: str):
        """Checks for new comments on posts and replies thoughtfully."""
        handler = self.handlers.get(platform)
        if not handler:
            log_event(f"No handler found for platform: {platform}", level='WARNING')
            return

        log_event(f"Checking for new comments on {platform}...", level='INFO')
        posts = await self.loop.run_in_executor(None, get_own_posts)
        for post_record in posts:
            post_uri = post_record.uri
            post_text = post_record.value.text

            comments = await self.loop.run_in_executor(None, lambda: get_comments_for_post(post_uri))
            for comment_thread in comments:
                comment = comment_thread.post
                comment_uri = comment.uri

                if comment_uri in self.processed_comments:
                    continue

                # Assuming the handler's client has a 'me' attribute with a 'did'
                if comment.author.did == handler.client.me.did:
                    self.processed_comments.add(comment_uri)
                    continue

                comment_text = comment.record.text
                reply_text = await self.engine.run_reply_generation(post_text, comment_text)

                if reply_text and "no" not in reply_text.lower():
                    log_event(f"Replying to {comment_uri}: {reply_text}", level='INFO')
                    payload = {
                        'action': 'reply',
                        'platform_identifier': platform,
                        'content': reply_text,
                        'root_uri': post_uri,
                        'parent_uri': comment_uri
                    }
                    await dispatch_structured_payload(payload, handler)

                self.processed_comments.add(comment_uri)

    async def run(self):
        """The main loop for the social media agent."""
        log_event("Social Media Agent started.", level='INFO')
        last_post_time = 0
        last_comment_check_time = 0
        post_interval = 600  # 10 minutes
        comment_check_interval = 300  # 5 minutes

        while True:
            try:
                current_time = time.time()
                platform = 'bluesky'

                # Check if it's time to post new content
                if current_time - last_post_time >= post_interval:
                    # Check if we're in cooldown period after a failure
                    if current_time - self.last_post_failure_time < self.post_failure_cooldown:
                        log_event(f"Skipping post due to recent failure. Cooldown remaining: {int(self.post_failure_cooldown - (current_time - self.last_post_failure_time))}s", level='INFO')
                    else:
                        log_event("Scheduled post time reached. Attempting to post.", level='INFO')
                        if platform in self.handlers:
                            await self._attempt_action(self._post_new_content, platform)
                        last_post_time = time.time()

                # Check if it's time to check for comments
                if current_time - last_comment_check_time >= comment_check_interval:
                    log_event("Scheduled comment check time reached. Attempting to check.", level='INFO')
                    if platform in self.handlers:
                        await self._attempt_action(self._check_and_reply_to_comments, platform)
                    last_comment_check_time = time.time()

                # Sleep for a short interval to prevent a busy loop and allow for graceful exit
                await asyncio.sleep(60)

            except Exception as e:
                log_event(f"Critical error in Social Media Agent loop: {e}\n{traceback.format_exc()}", level='CRITICAL')
                await asyncio.sleep(300)
