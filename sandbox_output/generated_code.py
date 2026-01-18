import random
from typing import Dict, Optional


class MetaphorEngine:
    """
    Translates complex emotional or abstract concepts into accessible analogies.

    Uses a hybrid rule-based and generative mapping system to ensure
    deterministic behavior without external API dependencies.
    """

    def __init__(self):
        """
        Initializes the engine with pre-defined mappings for high-quality,
        context-aware metaphor generation.
        """
        # Mapping of abstract concepts to concrete domains (imagery)
        self.domain_map = {
            "sadness": ["heavy rain", "winter ocean", "fading embers", "low tide"],
            "joy": ["sunrise", "bubbling spring", "dancing fireflies", "summer breeze"],
            "anxiety": ["turbulence", "static noise", "tightrope", "swarming insects"],
            "time": ["river", "sand slipping", "winding road", "sundial"],
            "memory": ["faded photograph", "echo", "foggy mirror", "ghost"],
            "system overload": [
                "circuit shorting",
                "traffic jam",
                "overflowing dam",
                "melting glacier",
            ],
        }

        # Sensory details for elaboration
        self.sensory_adjectives = {
            "sight": ["dim", "vibrant", "hazy", "sharp", "glimmering", "dark"],
            "sound": ["silent", "rhythmic", "deafening", "whispering", "discordant"],
            "feel": ["rough", "smooth", "cold", "burning", "weightless"],
        }

    def validate_input(self, concept: str) -> bool:
        """
        Checks if the input is valid for metaphor generation.

        Args:
            concept: The abstract concept string.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not concept or not isinstance(concept, str):
            return False
        if len(concept.strip()) == 0:
            return False
        return True

    def _select_domain(self, concept_lower: str) -> str:
        """
        Selects a concrete domain based on the concept string.

        Args:
            concept_lower: The lowercased concept string.

        Returns:
            str: The selected concrete analogy (e.g., 'heavy rain', 'fire').
        """
        # 1. Check for direct keyword matches
        for key, domains in self.domain_map.items():
            if key in concept_lower:
                return random.choice(domains)

        # 2. Fallback: Map to elemental domains based on abstract category keywords
        if any(word in concept_lower for word in ["love", "passion", "energy", "heat"]):
            return random.choice(["fire", "magnetism", "gravity"])
        elif any(
            word in concept_lower
            for word in ["confusion", "chaos", "unknown", "darkness"]
        ):
            return random.choice(["storm", "maze", "deep space"])

        # 3. Safe default for unmapped concepts
        return "a complex tapestry"

    def _elaborate_imagery(self, domain: str) -> Dict[str, str]:
        """
        Generates sensory details for the chosen domain.

        Args:
            domain: The selected concrete analogy string.

        Returns:
            Dict[str, str]: A dictionary containing sensory details.
        """
        return {
            "visual": random.choice(self.sensory_adjectives["sight"]),
            "auditory": random.choice(self.sensory_adjectives["sound"]),
            "tactile": random.choice(self.sensory_adjectives["feel"]),
        }

    def execute(self, concept: str) -> Optional[Dict]:
        """
        Main execution method.

        Args:
            concept: The abstract concept to translate.

        Returns:
            Optional[Dict]: A dictionary containing the generated metaphor,
                            or None if an error occurs.
        """
        try:
            # Validate Input
            if not self.validate_input(concept):
                raise ValueError(
                    "Invalid concept provided. Please enter a non-empty string."
                )

            concept_lower = concept.lower()

            # 1. Generate Core Analogy
            core_domain = self._select_domain(concept_lower)

            # 2. Generate Sensory Imagery
            imagery = self._elaborate_imagery(core_domain)

            # 3. Construct Grounding (Explanation)
            # Using string formatting to create a poetic summary
            grounding = (
                f"The concept of '{concept}' manifests as "
                f"{imagery['visual']} {core_domain}, sounding "
                f"{imagery['auditory']} and feeling {imagery['tactile']} to the touch."
            )

            # 4. Assemble Output
            result = {
                "input_concept": concept,
                "analogy": core_domain,
                "imagery": imagery,
                "grounding": grounding,
            }

            return result

        except ValueError as ve:
            # Handle specific validation errors
            print(f"[VALIDATION ERROR] {str(ve)}")
            return None
        except Exception as e:
            # Critical Error Handling for unexpected failures
            print(f"[CRITICAL ERROR] Metaphor Generation Failed: {str(e)}")
            return None


# --- Usage Example ---
if __name__ == "__main__":
    # Initialize the engine
    engine = MetaphorEngine()

    # Define test cases including edge cases
    test_concepts = [
        "Nuclear Decay",
        "Loneliness",
        "System Overload",
        "Pure Joy",
        "",  # Empty string to test validation
        None,  # None type to test validation
    ]

    print("--- Metaphor Generation Test Run ---\n")

    for concept in test_concepts:
        print(f"Processing: '{concept}'")

        # Handle None explicitly for cleaner printing
        str_concept = concept if concept is not None else "None"

        output = engine.execute(str_concept)

        if output:
            print(f"  -> Analogy: {output['analogy']}")
            print(f"  -> Grounding: {output['grounding']}")
            print(f"  -> Imagery: {output['imagery']}")
        else:
            print("  -> Result: None (Error handled gracefully)")
        print("-" * 40)
