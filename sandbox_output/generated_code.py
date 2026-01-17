"""
Metaphor Generation Module
Part of the generated_code.py execution environment.
This module provides the MetaphorEngine class for generating poetic metaphors
based on abstract concepts.
"""

import random
import logging
from typing import Optional

# Configure local logger (inherits from root if available)
logger = logging.getLogger(__name__)


class MetaphorEngine:
    """
    Generates metaphors by mapping abstract emotional concepts to concrete imagery.

    Attributes:
        concept_map (dict): Maps abstract concepts to lists of concrete domains.
        modifiers (list): Aesthetic modifiers to apply to the generated imagery.
    """

    def __init__(self):
        """
        Initializes the MetaphorEngine with internal datasets.

        Uses a fixed seed for reproducibility in generation tasks,
        though random.choice is used for variety.
        """
        random.seed(42)  # Fixed seed for consistent behavior

        # Mapping of abstract concepts to concrete visual domains
        self.concept_map = {
            "love": ["architecture", "weather", "alchemy", "gardening"],
            "sadness": ["ocean depths", "empty rooms", "winter fog", "autumn leaves"],
            "joy": [
                "sunlight on water",
                "spring blossoms",
                "harmony",
                "bursting light",
            ],
            "confusion": [
                "a tangled knot",
                "a maze of mirrors",
                "static noise",
                "broken glass",
            ],
            "existential dread": [
                "vast space",
                "silent chambers",
                "endless corridors",
                "weight of stone",
            ],
            "fear": ["shadows", "holding breath", "tight ropes", "dark water"],
            "hope": [
                "distant stars",
                "seedlings in concrete",
                "morning dew",
                "a rising tide",
            ],
        }

        # Aesthetic modifiers to enhance the imagery
        self.modifiers = [
            "luminescent",
            "heavy",
            "ancient",
            "fragile",
            "radiant",
            "whispering",
            "infinite",
            "echoing",
            "vibrant",
            "serene",
        ]

    def _get_domain_mapping(self, concept: str) -> Optional[str]:
        """
        Maps an abstract concept to a concrete domain.

        Args:
            concept (str): The abstract concept to map (e.g., "love").

        Returns:
            Optional[str]: The concrete domain (e.g., "architecture") or None if not found.
        """
        # Normalize input to lowercase for matching
        normalized_concept = concept.lower().strip()

        # Direct match
        if normalized_concept in self.concept_map:
            return random.choice(self.concept_map[normalized_concept])

        # Fallback: generic domain based on concept length (simple heuristic)
        if len(normalized_concept) > 0:
            domains = ["a quiet space", "a shifting landscape", "a distant echo"]
            return random.choice(domains)

        return None

    def generate(self, concept: str, tone: str = "poetic") -> str:
        """
        Generates a metaphor based on the provided concept and tone.

        Args:
            concept (str): The abstract concept to visualize.
            tone (str): The stylistic tone (default: "poetic").

        Returns:
            str: The formatted metaphor string.

        Raises:
            ValueError: If concept is empty or invalid.
        """
        if not concept or not isinstance(concept, str):
            raise ValueError("Concept must be a non-empty string.")

        # 1. Map concept to domain
        domain = self._get_domain_mapping(concept)
        if not domain:
            return "The concept remains elusive, like a shadow without form."

        # 2. Select random modifiers
        modifier = random.choice(self.modifiers)
        secondary_modifier = random.choice(self.modifiers)

        # Ensure modifiers are distinct for better flow
        while secondary_modifier == modifier and len(self.modifiers) > 1:
            secondary_modifier = random.choice(self.modifiers)

        # 3. Construct the metaphor based on tone
        if tone == "poetic":
            # Structure: [Concept] is like a [modifier] [domain], [secondary_modifier] in its nature.
            metaphor = f"{concept.capitalize()} is a {modifier} {domain}, {secondary_modifier} in its essence."
        elif tone == "dramatic":
            metaphor = f"Behold: {concept.capitalize()} is the {modifier} {domain}, and it is {secondary_modifier}!"
        elif tone == "minimal":
            metaphor = f"{concept.capitalize()}: {modifier} {domain}."
        else:
            # Default fallback
            metaphor = f"{concept.capitalize()} resembles a {modifier} {domain}."

        return format_output(metaphor)


def format_output(text: str) -> str:
    """
    Ensures the output string is grammatically correct and aesthetically pleasing.

    Args:
        text (str): The raw generated text.

    Returns:
        str: The formatted string with proper capitalization and punctuation.
    """
    if not text:
        return ""

    # Capitalize first letter
    text = text[0].upper() + text[1:]

    # Ensure it ends with punctuation (add period if missing standard punctuation)
    if not text.endswith((".", "!", "?")):
        text += "."

    return text


# ==========================================
# Integration for Task Execution
# ==========================================


def execute_metaphor_task(input_data: dict) -> dict:
    """
    Entry point for the metaphor generation task within generated_code.py.

    Expects input_data to contain:
        - 'concept': The abstract concept (str).
        - 'tone' (optional): The desired tone (str).

    Args:
        input_data (dict): Parameters for the task.

    Returns:
        dict: Result dictionary containing 'status' and 'output'.
    """
    try:
        engine = MetaphorEngine()

        concept = input_data.get("concept")
        tone = input_data.get("tone", "poetic")

        if not concept:
            return {
                "status": "error",
                "message": "Missing required parameter: 'concept'",
            }

        metaphor = engine.generate(concept, tone)

        logger.info(f"Generated metaphor for '{concept}': {metaphor}")

        return {"status": "success", "output": metaphor}

    except Exception as e:
        logger.error(f"Error generating metaphor: {e}")
        return {"status": "error", "message": str(e)}


# ==========================================
# Self-Test / Validation Block
# ==========================================
if __name__ == "__main__":
    # Basic validation logic to ensure the module works standalone
    print("--- Metaphor Engine Self-Test ---")

    test_inputs = [
        {"concept": "love"},
        {"concept": "sadness", "tone": "dramatic"},
        {"concept": "existential dread", "tone": "minimal"},
        {"concept": ""},  # Edge case
    ]

    for test in test_inputs:
        result = execute_metaphor_task(test)
        if result["status"] == "success":
            print(f"OK: {result['output']}")
        else:
            print(f"ERR: {result['message']}")

    print("--- Test Complete ---")
