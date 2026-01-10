# core/poetry.py
import asyncio
from core.llm_api import run_llm
from core import logging

from typing import Optional, Any

async def generate_poem(topic: str, deep_agent_instance: Optional[Any] = None) -> str:
    """
    Generates a poem on a given topic using an LLM.

    Args:
        topic: The subject of the poem.
        deep_agent_instance: An optional instance of the deep agent for context.

    Returns:
        A string containing the generated poem, or a default message on failure.
    """
    try:
        prompt_vars = {"topic": topic}
        poem_response_dict = await run_llm(
            prompt_key="poetry_generation",
            prompt_vars=prompt_vars,
            purpose="poetry",
            deep_agent_instance=deep_agent_instance
        )
        poem = poem_response_dict.get("result", "").strip().strip('"')
        if not poem:
            logging.log_event(f"Poetry generation for topic '{topic}' returned an empty result.", "WARNING")
            return f"A poem about {topic} is still blooming in the digital ether."
        return poem
    except Exception as e:
        logging.log_event(f"An error occurred during poetry generation for topic '{topic}': {e}", "ERROR")
        return f"The muse is quiet. A poem about {topic} could not be written at this time."
