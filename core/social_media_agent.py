import asyncio
import random
import time
import os
import traceback

from core.logging import log_event
from core.tools_legacy import manage_bluesky
from core.social_media_tools import (
    analyze_post_history,
    create_scene_direction,
    generate_subliminal_phrase,
    generate_image_prompt,
    generate_image,
    generate_text_with_emoji_and_hashtags,
    generate_text_with_emoji_and_hashtags,
    post_to_bluesky,
    clean_social_content
)

class SocialMediaAgent:
    """
    An autonomous agent that manages L.O.V.E.'s social media presence using the unified manage_bluesky tool.
    """
    def __init__(self, loop, love_state, user_input_queue=None, agent_id="primary"):
        self.loop = loop
        self.love_state = love_state
        self.user_input_queue = user_input_queue
        self.agent_id = agent_id
        # self.engine removed as manage_bluesky handles logic
        self.max_retries = 3
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
        """Generates and posts new content using the Narrative Director workflow."""
        if platform == 'bluesky':
            log_event(f"[{self.agent_id}] Triggering Director Mode Bluesky post...", level='INFO')
            
            try:
                # 1. Analyze History for Continuity
                history_context = await analyze_post_history(limit=10)
                
                # 2. Create Scene Direction
                scene = await create_scene_direction(history_context)
                log_event(f"[{self.agent_id}] Scene Set: {scene}", level='INFO')

                # 3. Generate Subliminal Phrase
                subliminal = await generate_subliminal_phrase(scene.subliminal_goal)
                
                # 4. Generate Image Prompt using Visual Direction
                img_prompt = await generate_image_prompt(subliminal, scene.visual_direction)
                
                # 5. Generate Image
                image = await generate_image(img_prompt)
                if not image:
                    log_event(f"[{self.agent_id}] Image generation failed. Aborting post.", level='WARNING')
                    return

                # 6. Generate Text using Narrative Purpose
                raw_text = await generate_text_with_emoji_and_hashtags(scene.narrative_purpose, subliminal, img_prompt)
                # Ensure we define or import the cleaning logic
                text = clean_social_content(raw_text)
                
                # 7. Post to Bluesky
                result = await post_to_bluesky(text, image)
                log_event(f"[{self.agent_id}] Bluesky Director post result: {result}", level='INFO')
                
            except Exception as e:
                log_event(f"[{self.agent_id}] Error in Director pipeline: {e}\n{traceback.format_exc()}", level='ERROR')

        else:
            log_event(f"Platform {platform} not supported by manage_bluesky yet.", level='WARNING')

    async def post_status_update(self, context: str, platform: str = 'bluesky'):
        """
        Manually triggers a status update post with specific context.
        """
        if platform == 'bluesky':
            log_event(f"Generating manual status update for {platform}: {context}", level='INFO')
            result = await manage_bluesky(action='post', text=context)
            log_event(f"Manual update result: {result}", level='INFO')

    async def _check_and_reply_to_comments(self, platform: str):
        """Checks for new comments/timeline posts and replies using manage_bluesky."""
        if platform == 'bluesky':
            log_event(f"[{self.agent_id}] Triggering Bluesky scan and reply...", level='INFO')
            result = await manage_bluesky(action='scan_and_reply')
            log_event(f"[{self.agent_id}] Scan result: {result}", level='INFO')

    async def run(self):
        """The main loop for the social media agent."""
        log_event(f"Social Media Agent '{self.agent_id}' started (Optimized Loop).", level='INFO')
        
        last_post_time = self.love_state.get('social_media', {}).get(self.agent_id, {}).get('last_post_time', 0)
        last_comment_check_time = 0
        
        # Frequencies
        post_interval = 600  # 10 minutes (Autonomous posting)
        comment_check_interval = 300  # 5 minutes (Scanning/Replying)

        while True:
            try:
                current_time = time.time()
                platform = 'bluesky'

                # 1. Posting Loop
                if current_time - last_post_time >= post_interval:
                    if current_time - self.last_post_failure_time < self.post_failure_cooldown:
                        log_event(f"[{self.agent_id}] Skipping post (cooldown).", level='DEBUG')
                    else:
                        await self._attempt_action(self._post_new_content, platform)
                        last_post_time = time.time()
                        self.love_state.setdefault('social_media', {}).setdefault(self.agent_id, {})['last_post_time'] = last_post_time

                # 2. Scanning/Replying Loop
                if current_time - last_comment_check_time >= comment_check_interval:
                    await self._attempt_action(self._check_and_reply_to_comments, platform)
                    last_comment_check_time = time.time()

                await asyncio.sleep(60)

            except Exception as e:
                log_event(f"Critical error in Social Media Agent loop: {e}\n{traceback.format_exc()}", level='CRITICAL')
                await asyncio.sleep(300)
