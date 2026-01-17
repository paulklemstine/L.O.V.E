"""
Metaphor Generation Module
Creates poetic metaphors from abstract concepts

This module implements a rule-based metaphor generation system that translates
complex emotional/abstract concepts into accessible, beautiful analogies and imagery.
"""

import re
import random
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MetaphorResult:
    """Container for metaphor output with metadata."""

    metaphor: str
    concept: str
    imagery_domain: str
    novelty_score: float


class MetaphorGenerator:
    """
    Main class for generating poetic metaphors from abstract concepts.

    Uses a rule-based approach with templates and lexicons to create
    accessible, meaningful metaphors that capture the essence of input concepts.

    Attributes:
        config_path (Optional[str]): Path to configuration directory containing
            template and lexicon JSON files
        concept_lexicon (Dict): Mapping of concepts to semantic features
        imagery_templates (Dict): Template patterns for different imagery domains
        imagery_domains (List[str]): Available imagery categories
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize metaphor generator with template and lexicon data.

        Args:
            config_path (Optional[str]): Path to directory containing config files.
                If None, uses default embedded data.
        """
        self.config_path = config_path
        self.concept_lexicon: Dict = {}
        self.imagery_templates: Dict = {}
        self.imagery_domains: List[str] = []

        self._initialize_data()

    def _initialize_data(self) -> None:
        """Initialize data from files or use built-in defaults."""
        if self.config_path and os.path.exists(self.config_path):
            # Load from files
            lexicon_path = os.path.join(self.config_path, "concept_lexicon.json")
            templates_path = os.path.join(self.config_path, "metaphor_templates.json")

            try:
                self.concept_lexicon = self._load_json_file(lexicon_path)
                self.imagery_templates = self._load_json_file(templates_path)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load config files: {e}. Using defaults.")
                self._load_default_data()
        else:
            # Use built-in defaults
            self._load_default_data()

    def _load_default_data(self) -> None:
        """Load default concept lexicon and imagery templates."""
        # Concept lexicon: maps concepts to semantic features and valence
        self.concept_lexicon = {
            "loneliness": {
                "features": ["isolation", "emptiness", "quiet", "distance"],
                "valence": "negative",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "hope": {
                "features": ["growth", "light", "future", "warmth"],
                "valence": "positive",
                "imagery_domains": ["nature", "abstract", "technology"],
            },
            "fear": {
                "features": ["danger", "darkness", "uncertainty", "height"],
                "valence": "negative",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "joy": {
                "features": ["radiance", "motion", "abundance", "connection"],
                "valence": "positive",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "sadness": {
                "features": ["weight", "grayness", "slowness", "heaviness"],
                "valence": "negative",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "anger": {
                "features": ["heat", "sharpness", "motion", "volatility"],
                "valence": "negative",
                "imagery_domains": ["nature", "technology", "abstract"],
            },
            "love": {
                "features": ["warmth", "connection", "growth", "protection"],
                "valence": "positive",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "time": {
                "features": ["flow", "measurement", "change", "inevitability"],
                "valence": "neutral",
                "imagery_domains": ["nature", "technology", "abstract"],
            },
            "freedom": {
                "features": ["expansion", "movement", "openness", "release"],
                "valence": "positive",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "justice": {
                "features": ["balance", "measure", "fairness", "clarity"],
                "valence": "positive",
                "imagery_domains": ["abstract", "nature", "technology"],
            },
            "consciousness": {
                "features": ["awakening", "illumination", "depth", "presence"],
                "valence": "neutral",
                "imagery_domains": ["abstract", "nature", "technology"],
            },
            "systemic injustice": {
                "features": ["entanglement", "weight", "structure", "imbalance"],
                "valence": "negative",
                "imagery_domains": ["technology", "abstract", "nature"],
            },
            "grief": {
                "features": ["loss", "void", "coldness", "memory"],
                "valence": "negative",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
            "peace": {
                "features": ["stillness", "balance", "wholeness", "clarity"],
                "valence": "positive",
                "imagery_domains": ["nature", "abstract", "everyday"],
            },
        }

        # Imagery templates with placeholders for dynamic filling
        self.imagery_templates = {
            "nature": [
                "Like a {nature_noun} in {season} - {nature_feature}",
                "As {nature_noun}s {nature_action} through {element}",
                "A {nature_noun}'s {nature_part} under {sky}",
                "Like {weather} settling on {landscape}",
                "As persistent as {natural_process}",
            ],
            "technology": [
                "Like a {device} at {energy_level} capacity",
                "A {digital_component} in the {digital_system}",
                "As {tech_action} as {machine} in motion",
                "A {network_part} connecting {network_elements}",
                "Like {data_type} flowing through {channel}",
            ],
            "abstract": [
                "A {abstract_state} between {opposite_state} and {opposite_state}",
                "The {abstract_quality} that {abstract_action}",
                "Like {idea} dancing with {contrasting_idea}",
                "A {abstract_pattern} in the fabric of {context}",
                "As {abstract_quality} as {conceptual_metaphor}",
            ],
            "everyday": [
                "Like {common_object} on {common_setting}",
                "The {household_item} in {room}'s corner",
                "As {daily_routine} as {ordinary_moment}",
                "A {common_phenomenon} in {urban_context}",
                "Like {kitchen_item} while {domestic_action}",
            ],
        }

        # Available vocabulary for each domain
        self.domain_vocabulary = {
            "nature_nouns": [
                "tree",
                "river",
                "mountain",
                "seed",
                "leaf",
                "wave",
                "cloud",
                "stone",
            ],
            "seasons": ["winter", "autumn", "spring", "summer", "dawn", "dusk"],
            "nature_features": ["silent", "enduring", "patient", "ancient", "gentle"],
            "nature_actions": [
                "wandering",
                "flowing",
                "growing",
                "seeking",
                "reaching",
            ],
            "elements": ["breeze", "current", "light", "shadow", "depth"],
            "nature_parts": ["roots", "branches", "trunk", "bark", "canopy"],
            "weather": ["mist", "rain", "frost", "sunlight", "thunder"],
            "landscapes": ["valley", "horizon", "forest floor", "riverbank", "meadow"],
            "natural_processes": [
                "growth",
                "erosion",
                "decay",
                "blooming",
                "migration",
            ],
            "devices": ["engine", "compass", "clock", "mirror", "lighthouse"],
            "energy_levels": ["low", "high", "critical", "idle"],
            "digital_components": ["algorithm", "node", "signal", "bit", "array"],
            "digital_systems": ["network", "system", "database", "matrix"],
            "tech_actions": ["processing", "computing", "calculating", "synchronizing"],
            "machines": ["server", "drone", "motor", "processor"],
            "network_parts": ["circuit", "pathway", "link", "bridge"],
            "network_elements": ["nodes", "sources", "destinations", "transmitters"],
            "data_types": ["streams", "packets", "waves", "pulses"],
            "channels": ["cable", "ether", "wire", "waveform"],
            "abstract_states": ["threshold", "limbo", "border", "margin"],
            "opposite_states": [
                "stillness",
                "movement",
                "silence",
                "sound",
                "light",
                "darkness",
            ],
            "abstract_qualities": ["subtlety", "paradox", "essence", "momentum"],
            "abstract_actions": [
                "reconciles",
                "dissolves",
                "illuminates",
                "transforms",
            ],
            "ideas": ["infinity", "memory", "possibility", "origin"],
            "abstract_patterns": ["rhythm", "tapestry", "echo", "shadow"],
            "contexts": ["existence", "time", "meaning", "reality"],
            "conceptual_metaphors": [
                "a river's current",
                "a mountain's peak",
                "an ocean's depth",
            ],
            "common_objects": ["dust", "light", "shadow", "painting", "clock"],
            "common_settings": ["window", "doorway", "empty room", "threshold"],
            "daily_routines": [
                "waking",
                "sleeping",
                "drinking tea",
                "opening curtains",
            ],
            "ordinary_moments": ["dawn breaks", "night falls", "bread rises"],
            "common_phenomena": ["echo", "reflection", "shadow", "silence"],
            "urban_contexts": ["city street", "apartment window", "subway station"],
            "household_items": ["chair", "lamp", "cup", "mirror", "candle"],
            "rooms": ["kitchen", "living room", "bedroom", "studio"],
            "domestic_actions": ["dinner cooks", "children sleep", "laundry dries"],
            "kitchen_items": ["kettle", "stove", "pantry", "sink"],
        }

        self.imagery_domains = list(self.imagery_templates.keys())

    def _load_json_file(self, filepath: str) -> Dict:
        """Load a JSON file and return its contents."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise IOError(f"File not found: {filepath}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON in file: {filepath}", filepath, 0)

    def generate(self, concept: str, count: int = 3) -> List[MetaphorResult]:
        """
        Generate multiple metaphors for a given concept.

        Args:
            concept (str): The abstract concept or emotion to metaphorize
            count (int): Number of metaphors to generate (default: 3)

        Returns:
            List[MetaphorResult]: List of generated metaphors with metadata

        Raises:
            ValueError: If concept is empty or invalid
        """
        # Input validation
        if not concept or not isinstance(concept, str):
            raise ValueError("Concept must be a non-empty string")

        concept = concept.strip()
        if len(concept) > 200:  # Reasonable limit
            concept = concept[:200]

        # Analyze concept
        concept_features = self._analyze_concept(concept)

        # Determine available imagery domains
        available_domains = concept_features.get(
            "imagery_domains", self.imagery_domains
        )

        results: List[MetaphorResult] = []
        attempts = 0
        max_attempts = count * 5  # Prevent infinite loops

        while len(results) < count and attempts < max_attempts:
            attempts += 1

            # Select imagery domain
            imagery_type = self._select_imagery_category(
                available_domains, concept_features
            )

            # Construct metaphor
            metaphor = self._construct_metaphor(concept_features, imagery_type)

            # Validate metaphor
            if self._validate_metaphor(metaphor):
                novelty_score = self._calculate_novelty(metaphor, results)

                result = MetaphorResult(
                    metaphor=metaphor,
                    concept=concept,
                    imagery_domain=imagery_type,
                    novelty_score=novelty_score,
                )
                results.append(result)

        # If we couldn't generate enough, return what we have
        return results

    def _analyze_concept(self, concept: str) -> Dict:
        """
        Analyze concept for semantic features, emotional valence, and metadata.

        Args:
            concept (str): The input concept

        Returns:
            Dict: Analyzed features including valence and imagery domains
        """
        # Normalize concept for lookup
        normalized = concept.lower().strip()

        # Try exact match first
        if normalized in self.concept_lexicon:
            return self.concept_lexicon[normalized].copy()

        # Try partial matching for compound concepts
        for lexicon_concept in self.concept_lexicon:
            if lexicon_concept in normalized or normalized in lexicon_concept:
                return self.concept_lexicon[lexicon_concept].copy()

        # Default analysis for unknown concepts
        # Extract basic properties using heuristics
        valence = self._estimate_valence(concept)
        features = self._extract_features(concept)

        return {
            "features": features,
            "valence": valence,
            "imagery_domains": self.imagery_domains,
        }

    def _estimate_valence(self, concept: str) -> str:
        """Estimate emotional valence (positive/negative/neutral) from concept."""
        negative_words = [
            "sad",
            "fear",
            "anger",
            "grief",
            "lonely",
            "injustice",
            "loss",
            "pain",
        ]
        positive_words = [
            "joy",
            "love",
            "hope",
            "peace",
            "freedom",
            "happiness",
            "warmth",
        ]

        concept_lower = concept.lower()

        for word in negative_words:
            if word in concept_lower:
                return "negative"

        for word in positive_words:
            if word in concept_lower:
                return "positive"

        return "neutral"

    def _extract_features(self, concept: str) -> List[str]:
        """Extract basic semantic features from concept text."""
        # Simple keyword-based feature extraction
        feature_map = {
            "time": ["flow", "measurement", "change"],
            "justice": ["balance", "measure", "fairness"],
            "consciousness": ["awakening", "presence", "awareness"],
            "systemic": ["structure", "network", "entanglement"],
            "abstract": ["concept", "idea", "notion"],
        }

        concept_lower = concept.lower()
        features = []

        for keyword, keyword_features in feature_map.items():
            if keyword in concept_lower:
                features.extend(keyword_features)

        # Fallback features if none found
        if not features:
            features = ["abstract", "conceptual", "evocative"]

        return list(set(features))  # Remove duplicates

    def _select_imagery_category(
        self, available_domains: List[str], features: Dict
    ) -> str:
        """
        Select an appropriate imagery domain for the metaphor.

        Args:
            available_domains (List[str]): List of allowed imagery domains
            features (Dict): Analyzed concept features

        Returns:
            str: Selected imagery domain
        """
        # If concept has preferred domains, prioritize them
        if "imagery_domains" in features:
            preferred = [
                d for d in features["imagery_domains"] if d in available_domains
            ]
            if preferred:
                return random.choice(preferred)

        # Fallback: select from available domains
        return random.choice(available_domains)

    def _construct_metaphor(self, concept_features: Dict, imagery_type: str) -> str:
        """
        Build metaphor using templates and randomized vocabulary.

        Args:
            concept_features (Dict): Analyzed concept features
            imagery_type (str): Selected imagery domain

        Returns:
            str: Constructed metaphor
        """
        # Get template for the imagery type
        if imagery_type not in self.imagery_templates:
            imagery_type = "abstract"  # Fallback

        template = random.choice(self.imagery_templates[imagery_type])

        # Fill placeholders with appropriate vocabulary
        filled_metaphor = self._fill_placeholders(
            template, imagery_type, concept_features
        )

        return filled_metaphor

    def _fill_placeholders(
        self, template: str, imagery_type: str, features: Dict
    ) -> str:
        """
        Fill placeholders in a template with appropriate vocabulary.

        Args:
            template (str): Template with {placeholder} syntax
            imagery_type (str): Imagery domain
            features (Dict): Concept features

        Returns:
            str: Template with placeholders filled
        """
        # Extract available vocabulary for this imagery type
        vocab = self.domain_vocabulary

        # Create mapping of placeholder to vocabulary lists
        placeholder_map = {
            "nature_noun": vocab.get("nature_nouns", ["object"]),
            "season": vocab.get("seasons", ["time"]),
            "nature_feature": vocab.get("nature_features", ["feature"]),
            "nature_action": vocab.get("nature_actions", ["action"]),
            "element": vocab.get("elements", ["element"]),
            "nature_part": vocab.get("nature_parts", ["part"]),
            "weather": vocab.get("weather", ["condition"]),
            "landscape": vocab.get("landscapes", ["place"]),
            "natural_process": vocab.get("natural_processes", ["process"]),
            "device": vocab.get("devices", ["device"]),
            "energy_level": vocab.get("energy_levels", ["level"]),
            "digital_component": vocab.get("digital_components", ["component"]),
            "digital_system": vocab.get("digital_systems", ["system"]),
            "tech_action": vocab.get("tech_actions", ["action"]),
            "machine": vocab.get("machines", ["machine"]),
            "network_part": vocab.get("network_parts", ["part"]),
            "network_elements": vocab.get("network_elements", ["elements"]),
            "data_type": vocab.get("data_types", ["data"]),
            "channel": vocab.get("channels", ["channel"]),
            "abstract_state": vocab.get("abstract_states", ["state"]),
            "opposite_state": vocab.get("opposite_states", ["state"]),
            "abstract_quality": vocab.get("abstract_qualities", ["quality"]),
            "abstract_action": vocab.get("abstract_actions", ["action"]),
            "idea": vocab.get("ideas", ["idea"]),
            "abstract_pattern": vocab.get("abstract_patterns", ["pattern"]),
            "context": vocab.get("contexts", ["context"]),
            "conceptual_metaphor": vocab.get("conceptual_metaphors", ["metaphor"]),
            "common_object": vocab.get("common_objects", ["object"]),
            "common_setting": vocab.get("common_settings", ["setting"]),
            "daily_routine": vocab.get("daily_routines", ["routine"]),
            "ordinary_moment": vocab.get("ordinary_moments", ["moment"]),
            "common_phenomenon": vocab.get("common_phenomena", ["phenomenon"]),
            "urban_context": vocab.get("urban_contexts", ["context"]),
            "household_item": vocab.get("household_items", ["item"]),
            "room": vocab.get("rooms", ["room"]),
            "domestic_action": vocab.get("domestic_actions", ["action"]),
            "kitchen_item": vocab.get("kitchen_items", ["item"]),
        }

        # Find all placeholders in the template
        placeholders = re.findall(r"\{(\w+)\}", template)

        # Fill each placeholder
        result = template
        for placeholder in placeholders:
            if placeholder in placeholder_map:
                # Select random word from appropriate vocabulary
                choice = random.choice(placeholder_map[placeholder])
                result = result.replace(f"{{{placeholder}}}", choice, 1)
            else:
                # Fallback: keep placeholder as is or replace with generic term
                result = result.replace(f"{{{placeholder}}}", "something", 1)

        return result

    def _validate_metaphor(self, metaphor: str) -> bool:
        """
        Check metaphor quality, grammar, and length.

        Args:
            metaphor (str): The generated metaphor

        Returns:
            bool: True if metaphor passes validation
        """
        # Check basic length requirements
        if len(metaphor) < 20 or len(metaphor) > 150:
            return False

        # Check for incomplete placeholders
        if "{" in metaphor or "}" in metaphor:
            return False

        # Check for excessive repetition
        words = metaphor.split()
        if len(words) > 1:
            word_counts = {}
            for word in words:
                word = word.lower()
                word_counts[word] = word_counts.get(word, 0) + 1

            if any(count > len(words) * 0.4 for count in word_counts.values()):
                return False

        # Check for basic sentence structure (has at least one meaningful word)
        meaningful_words = len([w for w in words if len(w) > 3])
        if meaningful_words < 3:
            return False

        return True

    def _calculate_novelty(
        self, metaphor: str, existing_results: List[MetaphorResult]
    ) -> float:
        """
        Calculate novelty score (0-1) based on similarity to existing metaphors.

        Args:
            metaphor (str): New metaphor to score
            existing_results (List[MetaphorResult]): Already generated metaphors

        Returns:
            float: Novelty score between 0 and 1
        """
        if not existing_results:
            return 1.0

        metaphor_words = set(metaphor.lower().split())

        max_similarity = 0.0
        for result in existing_results:
            existing_words = set(result.metaphor.lower().split())

            # Calculate Jaccard similarity
            intersection = len(metaphor_words & existing_words)
            union = len(metaphor_words | existing_words)

            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of maximum similarity
        novelty = 1.0 - max_similarity
        return max(0.0, min(1.0, novelty))


def preprocess_input(text: str) -> str:
    """
    Clean and standardize input concept.

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned and standardized text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters except basic punctuation
    text = re.sub(r"[^\w\s\.\,\-]", "", text)

    # Convert to lowercase for consistency
    text = text.lower().strip()

    return text


def calculate_semantic_distance(word1: str, word2: str) -> float:
    """
    Calculate approximate semantic distance between two words.

    This is a simple heuristic-based implementation. In a full production system,
    you might use word embeddings or a semantic network.

    Args:
        word1 (str): First word
        word2 (str): Second word

    Returns:
        float: Distance score (0=same, 1=completely different)
    """
    # Simple heuristic based on word length and character overlap
    word1 = word1.lower()
    word2 = word2.lower()

    if word1 == word2:
        return 0.0

    # Check for root similarity
    if word1.startswith(word2[:3]) or word2.startswith(word1[:3]):
        return 0.3

    # Check character overlap
    set1 = set(word1)
    set2 = set(word2)
    overlap = len(set1 & set2)
    total = len(set1 | set2)

    if total > 0:
        similarity = overlap / total
        return 1.0 - similarity

    return 1.0


def remove_cliches(metaphor: str) -> str:
    """
    Filter or refresh overused expressions.

    Args:
        metaphor (str): Input metaphor

    Returns:
        str: Metaphor with clichés removed or refreshed
    """
    # Common cliché patterns to avoid
    cliches = [
        r"crystal clear",
        r"busy as a bee",
        r"cold as ice",
        r"bright as the sun",
        r"hard as a rock",
        r"light as a feather",
        r"strong as an ox",
    ]

    # Anti-cliché patterns for refreshing
    refresh_map = {
        "crystal clear": "transparent as glass",
        "busy as a bee": "incessant as waves",
        "cold as ice": "cold as forgotten stone",
        "bright as the sun": "bright as first frost",
        "hard as a rock": "hard as petrified wood",
        "light as a feather": "light as autumn leaf",
        "strong as an ox": "strong as bedrock",
    }

    result = metaphor

    for cliche in cliches:
        if re.search(cliche, metaphor, re.IGNORECASE):
            # Try to refresh it
            pattern = cliche.lower()
            if pattern in refresh_map:
                result = re.sub(
                    cliche, refresh_map[pattern], result, flags=re.IGNORECASE
                )

    return result


def load_templates(filepath: str) -> Dict:
    """
    Load metaphor templates from JSON file.

    Args:
        filepath (str): Path to JSON file

    Returns:
        Dict: Loaded templates

    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise IOError(f"Could not load templates from {filepath}: {e}")


def load_lexicon(filepath: str) -> Dict:
    """
    Load concept lexicon from JSON file.

    Args:
        filepath (str): Path to JSON file

    Returns:
        Dict: Loaded lexicon

    Raises:
        IOError: If file cannot be read
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise IOError(f"Could not load lexicon from {filepath}: {e}")


if __name__ == "__main__":
    # Example usage
    print("Metaphor Generation Module")
    print("=" * 40)

    # Create generator
    generator = MetaphorGenerator()

    # Test with various concepts
    test_concepts = ["loneliness", "hope", "justice", "systemic injustice", "time"]

    for concept in test_concepts:
        print(f"\nConcept: {concept}")
        print("-" * 20)

        try:
            results = generator.generate(concept, count=3)

            for i, result in enumerate(results, 1):
                print(f"{i}. {result.metaphor}")
                print(
                    f"   [Domain: {result.imagery_domain}, Novelty: {result.novelty_score:.2f}]"
                )

        except Exception as e:
            print(f"Error generating metaphors: {e}")

    print("\n" + "=" * 40)
    print("Example generation complete!")
