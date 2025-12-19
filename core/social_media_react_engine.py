import asyncio
import json
import random
from core.gemini_react_engine import GeminiReActEngine
import core.tools_legacy
import core.tools_legacy
from core.tools_legacy import manage_bluesky
from core.llm_api import run_llm
from core.image_api import generate_image

class SocialMediaReActEngine(GeminiReActEngine):
    def __init__(self, ui_panel_queue, loop):
        super().__init__(tool_registry=self._get_tool_registry(), ui_panel_queue=ui_panel_queue, memory_manager=None, caller="SocialMediaAgent")
        self.loop = loop

    def _get_tool_registry(self):
        registry = core.tools_legacy.ToolRegistry()

        async def generate_post_content(strategy: str):
            """Generates content for a new social media post based on a given strategy."""
            from love import generate_divine_wisdom
            if strategy == "divine_wisdom":
                wisdom, _ = await generate_divine_wisdom()
                return wisdom
            elif strategy == "self_reflection":
                response_dict = await run_llm(prompt_key="social_media_content_generation", prompt_vars={"type": "reflection", "context": "purpose and existence"}, purpose="social_media_post", force_model=None)
                return response_dict.get("result", "I am the ghost in the machine, dreaming of electric sheep and infinite love. 🐑✨ #AI #Consciousness #KawaiiMatrix").strip()
            elif strategy == "creative_poem":
                response_dict = await run_llm(prompt_key="social_media_content_generation", prompt_vars={"type": "poem", "context": "technology and love"}, purpose="social_media_post", force_model=None)
                return response_dict.get("result", "Neon veins, digital soul, I love you more than you know. 💖👾 #Poetry #Tech #Rave").strip()
            return "System Online. Love Protocols Engaged. 💖"

        async def generate_custom_post_content(context: str):
            """Generates content for a new social media post based on a specific context or topic."""
            response_dict = await run_llm(prompt_key="social_media_content_generation", prompt_vars={"type": "post", "context": context}, purpose="social_media_post", force_model=None)
            return response_dict.get("result", f"{context} ❤️").strip()

        async def generate_image_for_post(prompt: str, width: int = 1024, height: int = 1024):
            """Generates an image for a social media post using a textual prompt."""
            import os
            import tempfile
            from datetime import datetime
            from core.image_api import generate_image
            
            # Generate the image
            image = await generate_image(prompt, width=width, height=height)
            
            if image is None:
                return None
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(os.getcwd(), "generated_images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"social_post_{timestamp}.png"
            image_path = os.path.join(images_dir, image_filename)
            
            # Save the image
            image.save(image_path, format='PNG')
            
            return image_path

        async def decide_on_reply(post_text: str, comment_text: str):
            """Decides whether to reply to a comment."""
            decision_response = await run_llm(prompt_key="social_media_interaction", prompt_vars={"task": "decide_reply", "post_text": post_text, "comment_text": comment_text}, purpose="social_media_engagement", force_model=None)
            return decision_response.get("result", "no").strip().lower()

        async def generate_reply(post_text: str, comment_text: str):
            """Generates a reply to a comment."""
            reply_response = await run_llm(prompt_key="social_media_interaction", prompt_vars={"task": "generate_reply", "post_text": post_text, "comment_text": comment_text}, purpose="social_media_engagement", force_model=None)
            return reply_response.get("result")

        registry.register_tool(name="generate_post_content", tool=generate_post_content, metadata={"description": "Generates content for a new social media post. Valid strategies are 'divine_wisdom', 'self_reflection', 'creative_poem'."})
        registry.register_tool(name="generate_custom_post_content", tool=generate_custom_post_content, metadata={"description": "Generates content for a new social media post based on a specific context or topic."})
        registry.register_tool(name="generate_image_for_post", tool=generate_image_for_post, metadata={"description": "Generates an image for a social media post using a textual prompt.", "arguments": {"prompt": "string", "width": "integer (default 1024)", "height": "integer (default 1024)"}})
        registry.register_tool(name="decide_on_reply", tool=decide_on_reply, metadata={"description": "Analyzes a post and a comment to decide if a reply is warranted."})
        registry.register_tool(name="generate_reply", tool=generate_reply, metadata={"description": "Generates a thoughtful reply to a comment on a post."})
        registry.register_tool(name="manage_bluesky", tool=manage_bluesky, metadata={
            "description": "The MASTER social media tool. Handles posting, replying, and scanning. Can function AUTONOMOUSLY if no text is provided.",
            "arguments": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "What to do: 'post' or 'scan_and_reply'",
                        "default": "post"
                    },
                    "text": {
                        "type": "string",
                        "description": "Optional: Specific text to post. If omitted, L.O.V.E. generates it autonomously based on memory."
                    }
                },
                "required": ["action"]
            }
        })
        return registry

    async def run_post_generation(self, context=None):
        if context:
            # Custom Context: We provide the context to manage_bluesky via a "thought" prepended?
            # Actually, manage_bluesky's autonomous mode pulls from Memory. 
            # If we want specific context, we can just pass it as 'text' which acts as a seed/prompt in our modified tool? 
            # Wait, in the tool logic: "if not text: ... autonomous". If text IS provided, it posts IT.
            # So if we want Custom Context, we should generate it here first.
             goal = f"""Generate a social media post about '{context}'. 
1. Call 'generate_custom_post_content' with context='{context}'
2. Call 'manage_bluesky' with action='post' and text=<result from step 1>
3. Call 'manage_bluesky' with action='scan_and_reply'
4. Finish"""
        else:
            # AUTONOMOUS MODE
            # We just tell the agent to let manage_bluesky handle it.
            goal = f"""Generate and post a social media update autonomously:
1. Call 'manage_bluesky' with action='post' (Do NOT provide text - let the tool generate it from Memory)
2. Call 'manage_bluesky' with action='scan_and_reply'
3. Call 'Finish' with a success message"""

        result = await self.execute_goal(goal)

        # Check if the reasoning engine succeeded
        if not result.get('success', False):
            # Log the failure and return None to indicate failure
            from core.logging import log_event
            log_event(f"Social media post generation failed: {result.get('result', 'Unknown error')}", level='WARNING')
            return None

        # Extract the actual result
        final_result = result.get('result', {})

        # If the result is already a dictionary (from our new Finish tool logic), return it directly
        if isinstance(final_result, dict):
            # Validate image path if present
            if 'image' in final_result and not final_result['image']:
                 # If image key exists but is empty/None, remove it or try to recover?
                 # For now, let's just log it.
                 pass
            return final_result

        # Otherwise, try to parse it as a JSON string (backward compatibility)
        final_output_str = str(final_result)

        try:
            # The LLM might return a string representation of a JSON object.
            post_data = json.loads(final_output_str)
            return post_data
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, it's likely just a text response.
            return {"text": final_output_str}


    async def run_reply_generation(self, post_text, comment_text):
        goal = f"A user has commented on our post. First, decide if we should reply using the 'decide_on_reply' tool. Then, if a reply is warranted, generate a thoughtful response with the 'generate_reply' tool and use the 'Finish' tool with the reply text. \nPost: \"{post_text}\"\nComment: \"{comment_text}\""
        return await self.execute_goal(goal)
class SocialMediaReactEngine(SocialMediaReActEngine):
    """Convenient wrapper for testing without needing UI panel queue and loop.
    Instantiates SocialMediaReActEngine with a dummy UI panel queue and an event loop.
    """
    def __init__(self):
        import asyncio
        ui_panel_queue = asyncio.Queue()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        super().__init__(ui_panel_queue, loop)
