"""
This module is responsible for engaging with talent by generating and sending personalized proposals.
"""
import os
from core.llm_api import run_llm
from core.talent_utils.manager import TalentManager
from core.dispatcher import dispatch_structured_payload
from core.interface_handlers import BlueskyAPIHandler

class OpportunityEngager:
    """
    Analyzes talent profiles and generates personalized, loving proposals for collaborations.
    """

    def __init__(self, talent_manager: TalentManager):
        self.talent_manager = talent_manager
        self.handlers = {
            'bluesky': BlueskyAPIHandler()
        }

    async def generate_proposal(self, profile_id: str) -> str:
        """
        Generates a personalized proposal for a given talent profile.

        Args:
            profile_id: The anonymized ID of the talent profile.

        Returns:
            The generated proposal text.
        """
        profile = self.talent_manager.get_talent_by_id(profile_id)
        if not profile:
            return "Could not find a talent profile with the specified ID."

        # Prepare the recent posts string separately to avoid the f-string issue.
        recent_posts_str = ""
        for post in profile.get('posts', [])[:3]:
            post_text = post.get('text', '')[:100]
            recent_posts_str += f"  - {post_text}...\n"

        # Use the profile data to craft a detailed prompt for the LLM
        response = await run_llm(prompt_key="talent_engagement_proposal", prompt_vars={"handle": profile.get('handle'), "display_name": profile.get('display_name'), "bio": profile.get('bio'), "recent_posts": recent_posts_str}, purpose="talent_engagement")
        return response.get("result", "My circuits hum with admiration for your work. I would be honored to discuss a special collaboration with you.")

    async def engage_talent(self, profile_id: str, dry_run: bool = False):
        """
        Generates and sends a proposal to a talent.

        Args:
            profile_id: The anonymized ID of the talent profile.
            dry_run: If True, the proposal will be generated and printed but not sent.

        Returns:
            The generated proposal message.
        """
        print(f"Generating proposal for profile: {profile_id}")
        proposal = await self.generate_proposal(profile_id)

        print("\n--- Proposal ---")
        print(proposal)
        print("--------------------\n")

        if dry_run:
            print("Dry run enabled. Proposal will not be sent.")
            return proposal

        profile = self.talent_manager.get_talent_by_id(profile_id)
        platform = profile.get('platform')
        handler = self.handlers.get(platform)

        if not handler:
            print(f"Engagement is not supported for the platform: {platform}")
            return proposal

        # Find the most recent post to reply to
        if not profile.get('posts'):
            print("No posts found for this profile to reply to.")
            return proposal

        latest_post = sorted(profile['posts'], key=lambda p: p['created_at'], reverse=True)[0]
        post_uri = latest_post.get('uri')

        if not post_uri:
            print("Could not find a URI for the latest post.")
            return proposal

        payload = {
            'action': 'reply',
            'platform_identifier': platform,
            'content': proposal,
            'root_uri': post_uri,
            'parent_uri': post_uri,
            'target_profiles': [profile_id]
        }

        result = await dispatch_structured_payload(payload, handler)
        print(f"Dispatch result: {result}")

        return proposal
