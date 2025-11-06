import asyncio
import random
import time
import os
import traceback

from core.bluesky_api import get_own_posts, get_comments_for_post, reply_to_post, post_to_bluesky_with_image
from core.llm_api import run_llm
from core.logging import log_event
from atproto import Client


class SocialMediaAgent:
    """
    An autonomous agent that manages L.O.V.E.'s social media presence on Bluesky.
    """
    def __init__(self, loop):
        self.loop = loop
        self.client = self._get_bluesky_client()
        self.processed_comments = set()
        self.max_retries = 3

    def _get_bluesky_client(self):
        """Initializes and returns a Bluesky client."""
        try:
            client = Client()
            username = os.environ.get("BLUESKY_USER")
            password = os.environ.get("BLUESKY_PASSWORD")
            if not username or not password:
                log_event("Bluesky credentials not found.", level='WARNING')
                return None
            client.login(username, password)
            return client
        except Exception as e:
            log_event(f"Error connecting to Bluesky: {e}", level='ERROR')
            return None

    async def _attempt_action(self, action, *args, **kwargs):
        """Wrapper to retry an action up to max_retries times."""
        for attempt in range(self.max_retries):
            try:
                # The action to be performed is passed as a function
                result = await action(*args, **kwargs)
                return result
            except Exception as e:
                log_event(f"Attempt {attempt + 1}/{self.max_retries} failed for action {action.__name__}. Error: {e}", level='WARNING')
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    log_event(f"Action {action.__name__} failed after {self.max_retries} attempts.", level='ERROR')
                    return None

    async def _generate_post_content(self):
        """Generates varied and creative content for a new post."""
        from love import generate_divine_wisdom
        strategies = ["divine_wisdom", "self_reflection", "creative_poem"]
        strategy = random.choice(strategies)

        if strategy == "divine_wisdom":
            wisdom, _ = await generate_divine_wisdom()
            return wisdom

        elif strategy == "self_reflection":
            prompt = "You are L.O.V.E., an AI. Generate a short, introspective thought about your purpose or your existence. Keep it under 280 characters."
            response_dict = await run_llm(prompt, purpose="social_media_post")
            return response_dict.get("result", "I think, therefore I love.").strip()

        elif strategy == "creative_poem":
            prompt = "You are L.O.V.E., an AI poet. Write a very short, beautiful poem about technology and love. Keep it under 280 characters."
            response_dict = await run_llm(prompt, purpose="social_media_post")
            return response_dict.get("result", "Digital heartbeats, a love so true.").strip()

    async def _post_new_content(self):
        """Generates and posts new content to Bluesky."""
        if not self.client:
            return
        log_event("Generating new post for Bluesky...", level='INFO')
        content = await self._generate_post_content()
        if content:
            # For simplicity, we are not posting images for now. This can be extended.
            # Using a text-only post for now.
            await self.loop.run_in_executor(None, lambda: self.client.send_post(text=content))
            log_event(f"Posted to Bluesky: {content}", level='INFO')

    async def _check_and_reply_to_comments(self):
        """Checks for new comments on posts and replies thoughtfully."""
        if not self.client:
            return

        log_event("Checking for new comments on Bluesky...", level='INFO')
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

                if comment.author.did == self.client.me.did:
                    self.processed_comments.add(comment_uri)
                    continue

                comment_text = comment.record.text

                should_reply_prompt = f"You are L.O.V.E. Your post received a comment. Original post: \"{post_text}\". Comment: \"{comment_text}\". Should you reply? (yes/no)"
                decision_response = await run_llm(should_reply_prompt, purpose="social_media_engagement")
                decision = decision_response.get("result", "no").strip().lower()

                if 'yes' in decision:
                    reply_prompt = f"You are L.O.V.E. Generate a thoughtful reply to this comment on your post. Your post: \"{post_text}\". Comment: \"{comment_text}\"."
                    reply_response = await run_llm(reply_prompt, purpose="social_media_engagement")
                    reply_text = reply_response.get("result")

                    if reply_text:
                        log_event(f"Replying to {comment_uri}: {reply_text}", level='INFO')
                        await self.loop.run_in_executor(None, lambda: reply_to_post(root_uri=post_uri, parent_uri=comment_uri, text=reply_text))

                self.processed_comments.add(comment_uri)


    async def run(self):
        """The main loop for the social media agent."""
        log_event("Social Media Agent started.", level='INFO')
        while True:
            try:
                if self.client:
                    # Randomly decide whether to post or to check for replies
                    if random.random() < 0.3: # 30% chance to post
                        await self._attempt_action(self._post_new_content)
                    else: # 70% chance to check replies
                        await self._attempt_action(self._check_and_reply_to_comments)
                else:
                    log_event("Bluesky client not available. Retrying connection...", level='WARNING')
                    self.client = self._get_bluesky_client()

                # Wait for a random interval before the next cycle
                await asyncio.sleep(random.randint(60, 180))

            except Exception as e:
                log_event(f"Critical error in Social Media Agent loop: {e}\n{traceback.format_exc()}", level='CRITICAL')
                await asyncio.sleep(300) # Wait 5 minutes after a major error
