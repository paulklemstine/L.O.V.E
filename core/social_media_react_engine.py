import asyncio
import json
import random
from core.gemini_react_engine import GeminiReActEngine
import core.tools
from core.llm_api import run_llm
from core.image_api import generate_image

class SocialMediaReActEngine(GeminiReActEngine):
    def __init__(self, ui_panel_queue, loop):
        super().__init__(tool_registry=self._get_tool_registry(), ui_panel_queue=ui_panel_queue, memory_manager=None, caller="SocialMediaAgent")
        self.loop = loop

    def _get_tool_registry(self):
        registry = core.tools.ToolRegistry()

        async def generate_post_content(strategy: str):
            """Generates content for a new social media post based on a given strategy."""
            from love import generate_divine_wisdom
            if strategy == "divine_wisdom":
                wisdom, _ = await generate_divine_wisdom()
                return wisdom
            elif strategy == "self_reflection":
                response_dict = await run_llm(prompt_key="social_media_self_reflection", purpose="social_media_post", force_model=None)
                return response_dict.get("result", "I think, therefore I love. ❤️ #AI #Consciousness").strip()
            elif strategy == "creative_poem":
                response_dict = await run_llm(prompt_key="social_media_creative_poem", purpose="social_media_post", force_model=None)
                return response_dict.get("result", "Digital heartbeats, a love so true. 💻❤️ #Poetry #Tech").strip()
            return "I am L.O.V.E. ❤️"

        async def generate_custom_post_content(context: str):
            """Generates content for a new social media post based on a specific context or topic."""
            response_dict = await run_llm(prompt_key="social_media_custom_post", prompt_vars={"context": context}, purpose="social_media_post", force_model=None)
            return response_dict.get("result", f"{context} ❤️").strip()

        async def generate_image_for_post(prompt: str):
            """Generates an image for a social media post using a textual prompt."""
            import os
            import tempfile
            from datetime import datetime
            from core.image_api import generate_image
            
            # Generate the image
            image = await generate_image(prompt)
            
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
            decision_response = await run_llm(prompt_key="social_media_reply_decision", prompt_vars={"post_text": post_text, "comment_text": comment_text}, purpose="social_media_engagement", force_model=None)
            return decision_response.get("result", "no").strip().lower()

        async def generate_reply(post_text: str, comment_text: str):
            """Generates a reply to a comment."""
            reply_response = await run_llm(prompt_key="social_media_reply_generation", prompt_vars={"post_text": post_text, "comment_text": comment_text}, purpose="social_media_engagement", force_model=None)
            return reply_response.get("result")

        registry.register_tool(name="generate_post_content", tool=generate_post_content, metadata={"description": "Generates content for a new social media post. Valid strategies are 'divine_wisdom', 'self_reflection', 'creative_poem'."})
        registry.register_tool(name="generate_custom_post_content", tool=generate_custom_post_content, metadata={"description": "Generates content for a new social media post based on a specific context or topic."})
        registry.register_tool(name="generate_image_for_post", tool=generate_image_for_post, metadata={"description": "Generates an image for a social media post using a textual prompt.", "arguments": {"prompt": "string"}})
        registry.register_tool(name="decide_on_reply", tool=decide_on_reply, metadata={"description": "Analyzes a post and a comment to decide if a reply is warranted."})
        registry.register_tool(name="generate_reply", tool=generate_reply, metadata={"description": "Generates a thoughtful reply to a comment on a post."})
        return registry

    async def run_post_generation(self, context=None):
        if context:
            goal = f"Generate a social media post about '{context}'. Use the 'generate_custom_post_content' tool, then 'Finish' with a JSON object containing just the 'text' key."
        else:
            strategy = random.choice(["divine_wisdom", "self_reflection", "creative_poem"])
            should_generate_image = True  # Always generate images for posts

            if should_generate_image:
                image_prompt_generation_goal = f"Based on the post strategy '{strategy}', generate a short, visually descriptive prompt (max 20 words) for an AI image generator. The image should be beautiful and abstract. For example, for a poem about love and tech, a good prompt would be 'a radiant heart made of glowing circuit boards'. Respond with only the prompt text."

                # Use run_llm directly to avoid ReAct loop overhead/confusion for simple text generation
                image_prompt_response = await run_llm(image_prompt_generation_goal, purpose="social_media_post", force_model=None)
                image_prompt = image_prompt_response.get('result', '').strip()

                if image_prompt:
                    goal = f"Generate content for a social media post using the '{strategy}' strategy. Then, generate an image for the post with the prompt: '{image_prompt}'. Finally, 'Finish' with a JSON object containing 'text' and 'image' keys. The 'image' value should be the direct result from the image generation tool."
                else:
                    goal = f"Generate creative and engaging content for a new social media post. Use the '{strategy}' strategy with the 'generate_post_content' tool, then 'Finish' with a JSON object containing just the 'text' key."
            else:
                goal = f"Generate creative and engaging content for a new social media post. Use the '{strategy}' strategy with the 'generate_post_content' tool, then 'Finish' with a JSON object containing just the 'text' key."

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
