import asyncio
import random
import traceback
from typing import Dict, Protocol

from core.logging import log_event
from core.tools import share_wisdom
from core.social_media_tools import (
    generate_post_concept,
    generate_image,
    post_to_bluesky,
    clean_social_content,
    analyze_post_history
)
from core.campaign_manager import Campaign
from core.story_manager import story_manager

class ChannelHandler(Protocol):
    """Protocol for channel handlers."""
    async def post_content(self, content: str, image: str = None) -> Dict:
        """Posts content to a specific channel."""
        ...

class BlueskyChannelHandler:
    """Handles posting content to Bluesky."""
    async def post_content(self, content: str, image: str = None) -> Dict:
        log_event(f"Posting to Bluesky: {content}", level='INFO')
        result = await post_to_bluesky(content, image)
        log_event(f"Bluesky post result: {result}", level='INFO')
        return result

class EmailChannelHandler:
    """Placeholder for handling email channels."""
    async def post_content(self, content: str, image: str = None) -> Dict:
        log_event(f"Simulating email post: {content}", level='INFO')
        # In a real implementation, this would use an email API
        await asyncio.sleep(1)
        return {"status": "success", "message": "Email sent successfully"}

class EngagementAgent:
    """
    An autonomous agent that manages omni-channel marketing campaigns.
    """
    WISDOM_POST_PROBABILITY = 0.33

    def __init__(self, loop, love_state, agent_id="primary"):
        self.loop = loop
        self.love_state = love_state
        self.agent_id = agent_id
        self.channel_handlers: Dict[str, ChannelHandler] = {
            "bluesky": BlueskyChannelHandler(),
            "email": EmailChannelHandler(),
        }

    async def _generate_content(self, campaign: Campaign):
        """Generates content for a campaign."""
        log_event(f"[{self.agent_id}] Generating content for campaign: {campaign.name}", level='INFO')
        try:
            if random.random() < self.WISDOM_POST_PROBABILITY:
                log_event(f"[{self.agent_id}] Choosing to share wisdom.", level='INFO')
                wisdom = await share_wisdom()
                image, _ = await generate_image("transcendent wisdom, universal love, digital art")
                return f"{wisdom} #UniversalLove #TranscendentKnowledge", image

            log_event(f"[{self.agent_id}] Triggering Director Mode post...", level='INFO')
            beat_data = story_manager.get_next_beat()
            recent_history = await analyze_post_history(limit=5)
            creator_goal = self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator.")
            concept = await generate_post_concept(beat_data, recent_history=recent_history, creator_goal=creator_goal)

            image, provider = await generate_image(concept.image_prompt, text_content=concept.subliminal_phrase)
            if not image:
                log_event(f"[{self.agent_id}] Image generation failed (Provider: {provider}). Aborting post.", level='WARNING')
                return None, None

            final_text = concept.post_text
            hashtags_to_add = [tag for tag in concept.hashtags if tag not in final_text]
            if hashtags_to_add:
                final_text += "\n" + " ".join(hashtags_to_add)

            return clean_social_content(final_text), image

        except Exception as e:
            log_event(f"[{self.agent_id}] Error in content generation: {e}\n{traceback.format_exc()}", level='ERROR')
            return None, None

    async def run_campaign(self, campaign: Campaign):
        """Runs a marketing campaign."""
        log_event(f"[{self.agent_id}] Running campaign: {campaign.name}", level='INFO')
        content, image = await self._generate_content(campaign)

        if not content:
            log_event(f"[{self.agent_id}] Failed to generate content for campaign: {campaign.name}", level='WARNING')
            return

        for channel in campaign.channels:
            handler = self.channel_handlers.get(channel)
            if handler:
                await handler.post_content(content, image)
            else:
                log_event(f"[{self.agent_id}] No handler for channel: {channel}", level='WARNING')
