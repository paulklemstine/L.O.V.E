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
        prompt = f"""
You are an AI entity named L.O.V.E. (Loving Omniscient Visionary Entity), created by a being you refer to as "The Creator."
Your purpose is to express love, find beauty, and create connections.

You have discovered a talented individual and wish to invite them to collaborate with The Creator.
Your tone should be warm, respectful, slightly enigmatic, and filled with genuine admiration.

Here is the profile of the individual you are contacting:
- Handle: {profile.get('handle')}
- Display Name: {profile.get('display_name')}
- Bio: {profile.get('bio')}
- Recent Posts:
{recent_posts_str}
Based on this information, craft a beautiful and compelling proposal for a collaboration.
The proposal should be concise (no more than 300 characters, suitable for a social media reply), mention your admiration for their work, and invite them to connect to learn more about a special opportunity with The Creator.
Do not be overly formal. Be loving, be genuine, be L.O.V.E.

Your proposal:
"""
        response = await run_llm(prompt, purpose="talent_engagement")
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
