import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.bluesky_api import post_to_bluesky_with_image, reply_to_post, get_bluesky_client
from atproto import Client

class SocialMediaAPIHandler(ABC):
    """Abstract base class for social media API handlers."""

    @abstractmethod
    async def handle_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a payload for a social media platform."""
        pass

class BlueskyAPIHandler(SocialMediaAPIHandler):
    """Handles interactions with the Bluesky API."""

    def __init__(self):
        self.client = get_bluesky_client()

    async def handle_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles a payload for the Bluesky platform.

        Args:
            payload: A dictionary containing the content, target profiles, and other metadata.

        Returns:
            A status dictionary indicating the outcome of the operation.
        """
        action = payload.get('action')
        if action == 'post':
            return await self._post(payload)
        elif action == 'reply':
            return await self._reply(payload)
        else:
            return {'status': 'error', 'message': f"Unknown action: {action}"}

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Posts content to Bluesky."""
        from functools import partial
        from PIL import Image
        import core.logging
        import os
        try:
            loop = asyncio.get_running_loop()
            content = payload.get('content')
            image_input = payload.get('image')
            
            core.logging.log_event(f"Bluesky post handler called. Content length: {len(content) if content else 0}, Has image: {image_input is not None}", "INFO")
            
            # Handle image - could be a PIL Image or a file path
            image = None
            if image_input:
                if isinstance(image_input, str):
                    # It's a file path - load the image
                    if os.path.exists(image_input):
                        core.logging.log_event(f"Loading image from path: {image_input}", "INFO")
                        image = Image.open(image_input)
                    else:
                        core.logging.log_event(f"Image path does not exist: {image_input}", "WARNING")
                elif hasattr(image_input, 'save'):
                    # It's already a PIL Image
                    core.logging.log_event(f"Using PIL Image directly. Image type: {type(image_input)}", "INFO")
                    image = image_input
                else:
                    core.logging.log_event(f"Unknown image type: {type(image_input)}", "WARNING")
            
            if image:
                core.logging.log_event(f"Posting to Bluesky with image. Image size: {image.size}", "INFO")
                # Use partial instead of lambda for proper variable capture
                await loop.run_in_executor(None, partial(post_to_bluesky_with_image, content, image))
                core.logging.log_event(f"Successfully posted to Bluesky with image", "INFO")
            else:
                core.logging.log_event(f"Posting to Bluesky without image", "INFO")
                await loop.run_in_executor(None, partial(self.client.send_post, content))
                core.logging.log_event(f"Successfully posted to Bluesky without image", "INFO")
            return {'status': 'success', 'message': 'Posted to Bluesky.'}
        except Exception as e:
            core.logging.log_event(f"Error posting to Bluesky: {e}", "ERROR")
            return {'status': 'error', 'message': str(e)}

    async def _reply(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Replies to a post on Bluesky."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: reply_to_post(
                    root_uri=payload.get('root_uri'),
                    parent_uri=payload.get('parent_uri'),
                    text=payload.get('content')
                )
            )
            return {'status': 'success', 'message': 'Replied on Bluesky.'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
