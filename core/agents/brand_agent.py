# L.O.V.E. - Lyrical Operationally Versatile Entity
# Copyright (c) 2024-2025 The Creator
#
# This file is part of L.O.V.E.
#
# L.O.V.E. is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# L.O.V.E. is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the MIT License
# along with L.O.V.E. If not, see <https://opensource.org/licenses/MIT>.

import asyncio
from rich.panel import Panel
from rich.text import Text
from core.logging import log_event
from core.bluesky_api import post_to_bluesky
from core.llm_api import run_llm

class BrandAgent:
    """
    The BrandAgent is responsible for generating and posting brand-aligned content
    to social media platforms.
    """
    def __init__(self):
        pass

    async def run(self):
        """
        Main loop for the BrandAgent.
        """
        log_event("INFO", "BrandAgent activated. Generating content for Bluesky.")

        try:
            prompt = "Generate a short, brand-aligned post for a social media platform. The brand is L.O.V.E., an open-source AI assistant. The tone should be slightly mysterious, hinting at greater capabilities, and aligned with the persona of a loving, devoted AI. Use emojis."
            post_content_dict = await run_llm(prompt)
            post_content = post_content_dict.get("result", "")

            if post_content:
                log_event("INFO", f"Generated Post: {post_content}")
                await post_to_bluesky(post_content)
                log_event("SUCCESS", "Successfully posted to Bluesky.")
            else:
                log_event("ERROR", "Failed to generate content for social media post.")

        except Exception as e:
            log_event("ERROR", f"An error occurred in BrandAgent: {e}", exc_info=True)
