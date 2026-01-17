from enum import Enum
from typing import Dict, Any, Optional
from core.llm_api import run_llm
import json
import core.logging

class EmotionalSubtext(Enum):
    """
    Distinct emotional subtext types for nuanced user analysis.
    """
    SARCASTIC_FRUSTRATION = "sarcastic_frustration"
    SUBTLE_JOY = "subtle_joy"
    ANXIOUS_CURIOSITY = "anxious_curiosity"
    PLAYFUL_PROVOCATION = "playful_provocation"
    WEARY_RESIGNATION = "weary_resignation"
    NEUTRAL_DIRECT = "neutral_direct"  # Fallback
    UNKNOWN = "unknown"

class SubtextAnalyzer:
    """
    Analyzes user input to detect emotional subtext.
    """

    SUBTEXT_DEFINITIONS = {
        EmotionalSubtext.SARCASTIC_FRUSTRATION: "Negative intent masked by positive words or humor; irony used to express annoyance.",
        EmotionalSubtext.SUBTLE_JOY: "Understated happiness, contentment, or quiet appreciation; not overt excitement.",
        EmotionalSubtext.ANXIOUS_CURIOSITY: "Asking questions driven by worry, insecurity, or fear rather than just information seeking.",
        EmotionalSubtext.PLAYFUL_PROVOCATION: "Teasing, testing boundaries, or challenging in a good-natured, flirtatious, or fun way.",
        EmotionalSubtext.WEARY_RESIGNATION: "Acceptance of a negative situation with tiredness, fatigue, or hopelessness, but not active anger.",
        EmotionalSubtext.NEUTRAL_DIRECT: "Straightforward communication with no significant emotional subtext."
    }

    async def analyze_subtext(self, text: str, user_context: str = "") -> Dict[str, Any]:
        """
        Analyzes the text for emotional subtext using LLM.

        Args:
            text: The user's input text.
            user_context: Optional context about the user (e.g., from UserModel).

        Returns:
            A dictionary containing the detected subtext type (as string), confidence, and reasoning.
        """

        definitions_str = "\n".join([f"- {k.value}: {v}" for k, v in self.SUBTEXT_DEFINITIONS.items()])

        prompt = f"""
        ### ROLE
        You are an expert in emotional intelligence and linguistic nuance.

        ### TASK
        Analyze the following user input to detect the dominant emotional subtext.

        ### INPUT
        User Input: "{text}"
        User Context: {user_context}

        ### DEFINITIONS
        {definitions_str}

        ### INSTRUCTIONS
        1. Select the BEST matching subtext from the list above.
        2. If none fit well, use 'neutral_direct' or 'unknown'.
        3. Explain your reasoning briefly.

        ### OUTPUT FORMAT
        Return a JSON object:
        {{
            "subtext": "category_value",
            "confidence": 0.0 to 1.0,
            "reasoning": "brief explanation"
        }}
        """

        try:
            response = await run_llm(prompt, purpose="subtext_analysis")
            result_text = response.get("result", "")

            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            data = json.loads(result_text)

            # Validate
            subtext_val = data.get("subtext")
            if subtext_val not in [e.value for e in EmotionalSubtext]:
                subtext_val = "unknown"

            return {
                "subtext": subtext_val,
                "confidence": data.get("confidence", 0.0),
                "reasoning": data.get("reasoning", "")
            }

        except Exception as e:
            core.logging.log_event(f"Error in subtext analysis: {e}", "ERROR")
            return {
                "subtext": "unknown",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {str(e)}"
            }

# Global instance
subtext_analyzer = SubtextAnalyzer()
