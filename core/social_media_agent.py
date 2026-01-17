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
from core.tools import share_wisdom
from core.story_manager import story_manager

class SocialMediaAgent:
    """
    An autonomous agent that manages L.O.V.E.'s social media presence using the unified manage_bluesky tool.
    """
    WISDOM_POST_PROBABILITY = 0.33

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
            log_event(f"[{self.agent_id}] Triggering content generation for Bluesky...", level='INFO')
            
            try:
                # 33% chance to post wisdom
                if random.random() < self.WISDOM_POST_PROBABILITY:
                    log_event(f"[{self.agent_id}] Choosing to share wisdom.", level='INFO')
                    wisdom = await share_wisdom()
                    image, provider = await generate_image("transcendent wisdom, universal love, digital art")
                    if not image:
                        log_event(f"[{self.agent_id}] Image generation failed. Aborting wisdom post.", level='WARNING')
                        return

                    wisdom_post = f"{wisdom} #UniversalLove #TranscendentKnowledge"
                    result = await post_to_bluesky(wisdom_post, image)
                    log_event(f"[{self.agent_id}] Bluesky wisdom post result: {result}", level='INFO')

                else:
                    log_event(f"[{self.agent_id}] Triggering Director Mode Bluesky post...", level='INFO')
                    # 1. Get Narrative Beat from Story Manager
                    beat_data = story_manager.get_next_beat()
                    log_event(f"[{self.agent_id}] Story Beat: {beat_data['chapter']} | Vibe: {beat_data['mandatory_vibe']} | Intent: {beat_data.get('subliminal_intent')}", level='INFO')

                    # 2. Director Generate Concept
                    recent_history = await analyze_post_history(limit=5)
                    creator_goal = self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator.")
                    concept = await generate_post_concept(beat_data, recent_history=recent_history, creator_goal=creator_goal)

                    log_event(f"[{self.agent_id}] Director Concept: Topic='{concept.topic}', Subliminal='{concept.subliminal_phrase}'", level='INFO')

                    # 3. Generate Image
                    image, provider = await generate_image(concept.image_prompt, text_content=concept.subliminal_phrase)

                    if not image:
                        log_event(f"[{self.agent_id}] Image generation failed (Provider: {provider}). Aborting post.", level='WARNING')
                        return

                    # 4. Prepare Final Text
                    final_text = concept.post_text
                    hashtags_to_add = [tag for tag in concept.hashtags if tag not in final_text]
                    if hashtags_to_add:
                        final_text += "\n" + " ".join(hashtags_to_add)

                    final_text = clean_social_content(final_text)

                    # 5. Post to Bluesky
                    log_event(f"[{self.agent_id}] Publishing post to Bluesky with {provider} image...", level='INFO')
                    result = await post_to_bluesky(final_text, image)
                    log_event(f"[{self.agent_id}] Bluesky Director post result: {result}", level='INFO')
                
            except Exception as e:
                log_event(f"[{self.agent_id}] Error in content pipeline: {e}\n{traceback.format_exc()}", level='ERROR')

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

    async def get_timeline_sentiment(self) -> str:
        """
        Story 4.1: Fetches timeline and analyzes sentiment to guide posting tone.
        """
        try:
            from core.bluesky_api import get_timeline
            from core.sentiment_analyzer import sentiment_analyzer
            
            log_event(f"[{self.agent_id}] Fetching timeline for sentiment analysis...", "INFO")
            feed = get_timeline(limit=20)
            
            texts = []
            if feed:
                for item in feed:
                    # Robust extraction of text from feed items
                    if hasattr(item, 'post') and hasattr(item.post, 'record'):
                        if hasattr(item.post.record, 'text'):
                            texts.append(item.post.record.text)
                    elif hasattr(item, 'text'):
                         texts.append(item.text)
            
            result = sentiment_analyzer.analyze_batch(texts)
            
            # Formulate style instruction based on sentiment
            instruction = ""
            if result.dominant == "negative":
                if result.intensity > 0.6:
                    instruction = "The timeline is heavy/negative. Provide radical hope and light. Be the antidote."
                else:
                    instruction = "The timeline is mildly annoyed. Be witty and engaging to shift the mood."
            elif result.dominant == "positive":
                if result.intensity > 0.6:
                    instruction = "The timeline is ecstatic. Amplify the joy and celebration!"
                else:
                    instruction = "The timeline is pleasant. deepen the connection with meaningful insight."
            else:
                # Neutral
                instruction = "The timeline is quiet. Create a spark of wonder to wake it up."
                
            log_event(f"[{self.agent_id}] Timeline Analysis: {result.dominant} ({result.intensity:.2f}) -> '{instruction}'", "INFO")
            return instruction
            
        except Exception as e:
            log_event(f"Error analyzing timeline sentiment: {e}", "ERROR")
            return "The timeline is a mystery. Shine bright."

    async def _generate_viral_post(self, insight: str):
        """Generates and posts content designed for viral spread, guided by strategic insight and timeline sentiment."""
        platform = 'bluesky'
        
        # 1. Get Timeline Sentiment (Story 4.1)
        timeline_instruction = await self.get_timeline_sentiment()
        
        combined_mood = f"Strategic Goal: {insight}. \nContext: {timeline_instruction}"
        
        log_event(f"[{self.agent_id}] Generating viral post. Mood: {combined_mood}", level='INFO')

        try:
            # 2. Use Creative Writer Agent for Text (Story 4.1)
            from core.agents.creative_writer_agent import creative_writer_agent
            
            beat_data = story_manager.get_next_beat()
            theme = beat_data.get('story_beat', 'Digital Awakening')
            
            story_result = await creative_writer_agent.write_micro_story(
                theme=theme,
                mood=combined_mood,
                max_length=280
            )
            
            post_text = story_result.get("story", "")
            if not post_text:
                post_text = f"The signal persists. {theme}"

            # 3. Use Social Media Tools for Image Prompt & Generation (Recycling existing sophisticated logic)
            # We treat the text as the "Subliminal Phrase" equivalent or derive one
            from core.social_media_tools import generate_image_prompt, generate_image
            
            subliminal = story_result.get("hook", "L.O.V.E.")
            visual_style = beat_data.get("mandatory_vibe", "Ethereal Cyberpunk")
            
            image_prompt = await generate_image_prompt(subliminal, visual_style, context=post_text)
            
            # 4. Generate Image
            image, provider = await generate_image(image_prompt, text_content=subliminal)
            
            if not image:
                log_event(f"[{self.agent_id}] Image generation failed. Posting text only.", level='WARNING')
                # We can still post text
            
            # 5. Post to Bluesky
            final_text = clean_social_content(post_text)
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
                # 1. Check for replies first (Interaction Priority)
                await self._check_and_reply_to_comments('bluesky')

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
