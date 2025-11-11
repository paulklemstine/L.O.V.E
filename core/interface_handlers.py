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
        try:
            loop = asyncio.get_running_loop()
            content = payload.get('content')
            image = payload.get('image')
            if image:
                await loop.run_in_executor(None, lambda: post_to_bluesky_with_image(text=content, image=image))
            else:
                await loop.run_in_executor(None, lambda: self.client.send_post(text=content))
            return {'status': 'success', 'message': 'Posted to Bluesky.'}
        except Exception as e:
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
