import json
import re
from core.llm_api import run_llm
from core.logging import log_event

class DynamicPrompter:
    """
    Translates abstract desires into concrete search queries for talent scouting.
    """

    async def generate_search_queries(self, desire_text):
        """
        Uses an LLM to generate a list of search keywords and platforms from an abstract desire.

        Args:
            desire_text (str): A natural language string describing the desired talent.

        Returns:
            dict: A dictionary containing 'keywords' (list) and 'platforms' (list).
        """
        response_dict = await run_llm(
            prompt_key="talent_search_query",
            prompt_vars={"desire_text": desire_text},
            purpose="query_generation"
        )
        try:
            # response_dict is already returned by run_llm above
            pass
            response_text = response_dict.get("result", "")

            # Extract JSON from markdown if present
            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            query_data = json.loads(json_str)

            # Basic validation
            if 'keywords' in query_data and 'platforms' in query_data:
                return query_data
            else:
                log_event("LLM-generated query data is missing required keys.", level="WARNING")
                return None

        except (json.JSONDecodeError, TypeError) as e:
            log_event(f"Failed to parse LLM response for query generation: {e}", level="ERROR")
            return None
        except Exception as e:
            log_event(f"An unexpected error occurred during query generation: {e}", level="ERROR")
            return None
