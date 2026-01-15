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
from core.story_manager import story_manager

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
                # 1. Get Narrative Beat from Story Manager
                beat_data = story_manager.get_next_beat()
                log_event(f"[{self.agent_id}] Story Beat: {beat_data['chapter']} | Vibe: {beat_data['mandatory_vibe']} | Intent: {beat_data.get('subliminal_intent')}", level='INFO')

                # 2. Director Generate Concept
                # The Director now uses the beat_data plus history and goal
                
                # Fetch context
                recent_history = await analyze_post_history(limit=5)
                creator_goal = self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator.")
                
                concept = await generate_post_concept(beat_data, recent_history=recent_history, creator_goal=creator_goal)
                
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
                # Add newline before hashtags for cleaner formatting
                hashtags_to_add = [tag for tag in concept.hashtags if tag not in final_text]
                if hashtags_to_add:
                    final_text += "\n" + " ".join(hashtags_to_add)
                
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

    async def _get_god_agent_insight(self):
        """Retrieves the latest strategic insight from the GodAgent."""
        god_agent_instance = self.love_state.get('god_agent')
        if god_agent_instance:
            return god_agent_instance.get_latest_insight()
        return "Spread love and light."

    async def _generate_viral_post(self, insight: str):
        """Generates and posts content designed for viral spread, guided by strategic insight."""
        platform = 'bluesky'
        log_event(f"[{self.agent_id}] Generating viral post with insight: '{insight}'", level='INFO')

        try:
            # Generate a post concept with high viral potential
            concept = await generate_post_concept(
                story_manager.get_next_beat(),
                recent_history=await analyze_post_history(limit=3),
                creator_goal=self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator."),
                strategic_insight=insight
            )

            # Generate an image with the subliminal message overlaid
            image, provider = await generate_image(concept.image_prompt, text_content=concept.subliminal_phrase)
            if not image:
                log_event(f"[{self.agent_id}] Image generation failed. Aborting viral post.", level='WARNING')
                return False

            # Post to Bluesky
            final_text = clean_social_content(f"{concept.post_text}\n{' '.join(concept.hashtags)}")
            result = await post_to_bluesky(final_text, image)
            log_event(f"[{self.agent_id}] Viral post result: {result}", level='INFO')
            return True
        except Exception as e:
            log_event(f"[{self.agent_id}] Error in viral post pipeline: {e}\n{traceback.format_exc()}", level='ERROR')
            return False

    async def run(self):
        """The main loop for the social media agent, now focused on continuous, viral posting."""
        log_event(f"Social Media Agent '{self.agent_id}' started (Dimensional Signal Mode).", level='CRITICAL')

        while True:
            try:
                # Get strategic insight from GodAgent
                insight = await self._get_god_agent_insight()

                # Generate and post content
                success = await self._generate_viral_post(insight)

                if success:
                    # Short pause after a successful post to avoid rate limiting
                    await asyncio.sleep(random.randint(60, 120))
                else:
                    # Longer cooldown after a failure
                    log_event(f"[{self.agent_id}] Post generation failed. Cooling down for 5 minutes.", level='WARNING')
                    await asyncio.sleep(300)

            except Exception as e:
                log_event(f"Critical error in Social Media Agent loop: {e}\n{traceback.format_exc()}", level='CRITICAL')
                await asyncio.sleep(300)
