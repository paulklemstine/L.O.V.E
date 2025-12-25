import asyncio
import random
import time
import os
import traceback

from core.logging import log_event
from core.tools_legacy import manage_bluesky
from core.social_media_tools import (
    analyze_post_history,
    generate_post_concept,
    generate_image,
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



from core.story_manager import story_manager

    async def _post_new_content(self, platform: str):
        """Generates and posts new content using the Narrative Director workflow."""
        if platform == 'bluesky':
            log_event(f"[{self.agent_id}] Triggering Director Mode Bluesky post...", level='INFO')
            
            try:
                # 1. Get Narrative Beat from Story Manager
                beat_data = story_manager.get_next_beat()
                log_event(f"[{self.agent_id}] Story Beat: {beat_data['chapter']} | Vibe: {beat_data['mandatory_vibe']}", level='INFO')

                # 2. Director Generate Concept
                # The Director now uses the beat_data directly
                concept = await generate_post_concept(beat_data)
                
                log_event(f"[{self.agent_id}] Director Concept: Topic='{concept.topic}', Subliminal='{concept.subliminal_phrase}'", level='INFO')

                # 3. Generate Image using Director's Visual Description
                # Pass subliminal phrase for overlay (Story 2.1 Fix: Always overlay text)
                image, provider = await generate_image(concept.image_prompt, text_content=concept.subliminal_phrase)
                
                # Story 3.1: Explicit Null Image Handling -> Abort
                if not image:
                    log_event(f"[{self.agent_id}] Image generation failed (Provider: {provider}). Aborting post.", level='WARNING')
                    return

                # 4. Prepare Final Text
                # The Director returns cleaned text, but we can double check or append hashtags if missing
                final_text = concept.post_text
                # Ensure hashtags are appended if they aren't in the text
                for tag in concept.hashtags:
                    # Simple check to avoid duplication (case-insensitive check would be better but this is MVP)
                    if tag not in final_text:
                        final_text += f" {tag}"
                
                final_text = clean_social_content(final_text)
                
                # 5. Post to Bluesky
                # Story 3.1: Log which provider was used for the final post
                log_event(f"[{self.agent_id}] Publishing post to Bluesky with {provider} image...", level='INFO')
                
                result = await post_to_bluesky(final_text, image)
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
