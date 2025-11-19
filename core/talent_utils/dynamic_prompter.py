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
        prompt = f"""
You are an AI assistant specialized in talent acquisition for a creative project.
Your task is to translate an abstract desire from The Creator into a structured set of search queries.

The Creator's Desire:
---
{desire_text}
---

Based on this desire, generate a JSON object with two keys:
1.  "keywords": A list of 3-5 concise, effective search keywords. These should be terms that would likely appear in a creative professional's bio or posts (e.g., "generative artist", "AI art", "runway model", "fashion design").
2.  "platforms": A list of 1-3 relevant social media platforms where such talent is likely to be found. Choose from: "bluesky", "instagram", "tiktok".

The output must be only the raw JSON object, with no other text or explanation.

Example Output:
```json
{{
  "keywords": ["fashion model", "couture", "runway", "vogue"],
  "platforms": ["instagram", "tiktok"]
}}
```
"""
        try:
            response_dict = await run_llm(prompt, purpose="query_generation")
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
