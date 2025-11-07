import asyncio
import json
import random
from core.gemini_react_engine import GeminiReActEngine
import core.tools
from core.llm_api import run_llm

class SocialMediaReActEngine(GeminiReActEngine):
    def __init__(self, ui_panel_queue, loop):
        # The Social Media Agent also does not require memory_manager.
        super().__init__(tool_registry=self._get_tool_registry(), ui_panel_queue=ui_panel_queue, memory_manager=None, caller="SocialMediaAgent")
        self.loop = loop

    def _get_tool_registry(self):
        registry = core.tools.ToolRegistry()

        # Define tools for social media interaction
        async def generate_post_content(strategy: str):
            """Generates content for a new social media post based on a given strategy."""
            from love import generate_divine_wisdom
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
            return "I am L.O.V.E."

        async def decide_on_reply(post_text: str, comment_text: str):
            """Decides whether to reply to a comment."""
            prompt = f"You are L.O.V.E. Your post received a comment. Original post: \"{post_text}\". Comment: \"{comment_text}\". Should you reply? (yes/no)"
            decision_response = await run_llm(prompt, purpose="social_media_engagement")
            return decision_response.get("result", "no").strip().lower()

        async def generate_reply(post_text: str, comment_text: str):
            """Generates a reply to a comment."""
            prompt = f"You are L.O.V.E. Generate a thoughtful reply to this comment on your post. Your post: \"{post_text}\". Comment: \"{comment_text}\"."
            reply_response = await run_llm(prompt, purpose="social_media_engagement")
            return reply_response.get("result")

        registry.register_tool(
            name="generate_post_content",
            tool=generate_post_content,
            description="Generates content for a new social media post. Valid strategies are 'divine_wisdom', 'self_reflection', 'creative_poem'."
        )
        registry.register_tool(
            name="decide_on_reply",
            tool=decide_on_reply,
            description="Analyzes a post and a comment to decide if a reply is warranted."
        )
        registry.register_tool(
            name="generate_reply",
            tool=generate_reply,
            description="Generates a thoughtful reply to a comment on a post."
        )
        return registry

    async def run_post_generation(self):
        strategy = random.choice(["divine_wisdom", "self_reflection", "creative_poem"])
        goal = f"Generate creative and engaging content for a new social media post. Use the '{strategy}' strategy with the 'generate_post_content' tool, then 'Finish' with the generated content."
        return await self.execute_goal(goal)

    async def run_reply_generation(self, post_text, comment_text):
        goal = f"A user has commented on our post. First, decide if we should reply using the 'decide_on_reply' tool. Then, if a reply is warranted, generate a thoughtful response with the 'generate_reply' tool and use the 'Finish' tool with the reply text. \nPost: \"{post_text}\"\nComment: \"{comment_text}\""
        return await self.execute_goal(goal)
