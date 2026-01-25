import json
import logging
from core.llm_api import run_llm

class MetaphorGenerator:
    """
    Generates conceptual metaphors based on input concepts and desired tones.
    """

    async def generate_metaphor(self, concept: str, tone: str) -> dict:
        """
        Generates metaphors for a given concept and tone.

        Args:
            concept (str): The target concept (e.g., "grief", "quantum entanglement").
            tone (str): The desired tone (e.g., "melancholy", "scientific").

        Returns:
            dict: A dictionary containing the concept analysis and generated metaphors.
        """
        prompt = self._construct_prompt(concept, tone)

        try:
            # We use 'creative_writing' purpose to encourage better quality
            response = await run_llm(prompt_text=prompt, purpose="creative_writing")

            if not response or not response.get("result"):
                logging.error("LLM returned empty response for metaphor generation.")
                return {"error": "Failed to generate metaphors."}

            result_text = response["result"]
            return self._parse_response(result_text)

        except Exception as e:
            logging.error(f"Error during metaphor generation: {e}")
            return {"error": str(e)}

    def _construct_prompt(self, concept: str, tone: str) -> str:
        return f"""
You are a Conceptual Translation Engine. Your task is to translate the abstract concept of "{concept}" into accessible, beautiful analogies or imagery, with a tone that is "{tone}".

Please perform the following steps:
1.  **Concept Mapping:** Analyze the input concept to identify core attributes, associated feelings, and structural relationships.
2.  **Analogy Generation:** Construct three distinct metaphorical representations.
3.  **Visual/Textual Output:** For each metaphor, provide a textual description and a visual prompt suitable for image generation.

**Quality Guardrails:**
- Avoid clichés unless explicitly requested.
- Avoid harmful stereotypes and logical inconsistencies.
- Ensure the metaphors are accessible to a general audience.

**Output Format:**
You must output strictly valid JSON with no markdown formatting or extra text. The structure should be as follows:

{{
  "concept_analysis": {{
    "core_attributes": ["attr1", "attr2"],
    "associated_feelings": ["feeling1", "feeling2"],
    "structural_relationships": ["rel1", "rel2"]
  }},
  "metaphors": [
    {{
      "id": 1,
      "textual_description": "Description of the first metaphor...",
      "visual_prompt": "A detailed visual prompt for generating an image of this metaphor..."
    }},
    {{
      "id": 2,
      "textual_description": "Description of the second metaphor...",
      "visual_prompt": "Visual prompt for the second metaphor..."
    }},
    {{
      "id": 3,
      "textual_description": "Description of the third metaphor...",
      "visual_prompt": "Visual prompt for the third metaphor..."
    }}
  ]
}}
"""

    def _parse_response(self, result_text: str) -> dict:
        """
        Parses the LLM response text into a dictionary.
        Handles potential markdown code blocks.
        """
        try:
            # clean up markdown code blocks if present
            cleaned_text = result_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]

            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            return json.loads(cleaned_text.strip())
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {result_text}")
            return {"error": "Invalid JSON response from LLM", "raw_response": result_text}
