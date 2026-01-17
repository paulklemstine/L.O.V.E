"""
Metaphor Generation Module
==========================

A creative module for translating complex emotional/abstract concepts into
accessible, beautiful analogies and imagery. This implementation uses a
rule-based template system with extensible NLP capabilities.

Features:
- Template-based metaphor generation
- Concept decomposition using keyword analysis
- Imagery enhancement for visual richness
- Basic safety filtering
- Configurable model type (template or llm placeholder)

Dependencies:
- NLTK (optional, for enhanced decomposition): pip install nltk
  Then run: python -c "import nltk; nltk.download('punkt')"
- Profanity-filter (optional): pip install profanity-filter
- Transformers (optional for LLM mode): pip install transformers torch

Security Notes:
- No external API calls by default
- No hardcoded secrets
- Safe subprocess usage (none required)
- Input validation and error handling
"""

import json
import logging
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports (handled gracefully)
try:
    import nltk
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.info("NLTK not available. Using fallback keyword analysis.")

try:
    from profanity_filter import ProfanityFilter

    PROFANITY_AVAILABLE = True
except ImportError:
    PROFANITY_AVAILABLE = False
    logger.info("Profanity filter not available. Using basic filtering.")

try:
    import torch
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers not available. LLM mode disabled.")


class Decomposer(ABC):
    """Abstract base class for concept decomposition strategies."""

    @abstractmethod
    def decompose(self, concept: str) -> Dict[str, Any]:
        """Extract attributes from a concept string."""
        pass


class KeywordDecomposer(Decomposer):
    """Keyword-based decomposer using pattern matching."""

    # Expanded keyword database for richer attribute extraction
    EMOTION_KEYWORDS = {
        "sadness": {
            "emotion": "loss",
            "element": "fog",
            "action": "shrouds",
            "context": "the mind",
        },
        "grief": {
            "emotion": "mourning",
            "element": "storm",
            "action": "erodes",
            "context": "the heart",
        },
        "joy": {
            "emotion": "delight",
            "element": "sunlight",
            "action": "illuminates",
            "context": "the world",
        },
        "anxiety": {
            "emotion": "worry",
            "element": "tangled vines",
            "action": "constricts",
            "context": "the chest",
        },
        "anger": {
            "emotion": "rage",
            "element": "fire",
            "action": "consumes",
            "context": "the soul",
        },
        "love": {
            "emotion": "affection",
            "element": "warm current",
            "action": "envelops",
            "context": "the being",
        },
        "fear": {
            "emotion": "dread",
            "element": "cold shadow",
            "action": "looms",
            "context": "the future",
        },
        "hope": {
            "emotion": "optimism",
            "element": "dawn light",
            "action": "breaks",
            "context": "the horizon",
        },
        "loneliness": {
            "emotion": "isolation",
            "element": "empty room",
            "action": "echoes",
            "context": "the silence",
        },
        "time": {
            "emotion": "transience",
            "element": "flowing river",
            "action": "carries",
            "context": "all things",
        },
        "infinity": {
            "emotion": "vastness",
            "element": "starry expanse",
            "action": "stretches",
            "context": "the void",
        },
    }

    def decompose(self, concept: str) -> Dict[str, Any]:
        """
        Decompose concept into attributes using keyword matching.

        Args:
            concept: Input string describing emotion or abstract concept

        Returns:
            Dictionary of attributes with fallback defaults

        Raises:
            ValueError: If concept is empty or too short
        """
        if not concept or not concept.strip():
            raise ValueError("Concept cannot be empty")

        # Normalize concept
        normalized = concept.lower().strip()

        # Check for exact match in keywords
        for key, attributes in self.EMOTION_KEYWORDS.items():
            if key in normalized:
                logger.info(f"Matched concept '{concept}' to keyword '{key}'")
                return attributes

        # Fallback: Extract first word and use generic mapping
        words = normalized.split()
        primary_concept = words[0] if words else "unknown"

        # Use primary concept if in keywords, otherwise default
        if primary_concept in self.EMOTION_KEYWORDS:
            return self.EMOTION_KEYWORDS[primary_concept]

        # Generic fallback
        return {
            "emotion": primary_concept,
            "element": "light",
            "action": "transforms",
            "context": "the moment",
        }


class NLTKDecomposer(Decomposer):
    """Enhanced decomposer using NLTK for tokenization and POS tagging."""

    def __init__(self):
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for NLTKDecomposer. Run: pip install nltk"
            )

        # Ensure required NLTK data is available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.warning("NLTK punkt not found. Downloading...")
            nltk.download("punkt")

    def decompose(self, concept: str) -> Dict[str, Any]:
        """
        Enhanced decomposition using NLTK tokenization and keyword analysis.

        Args:
            concept: Input string

        Returns:
            Dictionary of attributes
        """
        if not concept or not concept.strip():
            raise ValueError("Concept cannot be empty")

        # Tokenize and analyze
        tokens = word_tokenize(concept.lower())

        # Check against known emotions/concepts
        for key, attributes in KeywordDecomposer.EMOTION_KEYWORDS.items():
            if any(key in token for token in tokens):
                logger.info(f"NLTK matched concept '{concept}' to keyword '{key}'")
                return attributes

        # Use NLTK to identify main noun/verb (simplified)
        # In a full implementation, we'd use POS tagging
        primary = tokens[0] if tokens else "concept"

        # Map to template attributes
        return {
            "emotion": primary,
            "element": "wind",  # Nature-based default
            "action": "whispers",
            "context": "the vastness",
        }


class TemplateEngine:
    """Handles template selection and formatting."""

    def __init__(self):
        self.templates = self._load_builtin_templates()

    def _load_builtin_templates(self) -> List[str]:
        """Load built-in metaphor templates."""
        return [
            "{emotion} is like {element} that {action} {context}",
            "In the heart of {concept}, {emotion} feels like {element}",
            "{emotion} transforms into {element}, {action} through {context}",
            "The weight of {emotion} resembles {element} in {context}",
            "Like {element} in the dark, {emotion} {action} {context}",
            "{emotion} dances as {element}, {action} across {context}",
        ]

    def load_from_file(self, filepath: str) -> List[str]:
        """
        Load templates from a JSON file.

        Args:
            filepath: Path to JSON file containing template strings

        Returns:
            List of templates
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(t, str) for t in data):
                    logger.info(f"Loaded {len(data)} templates from {filepath}")
                    return data
                else:
                    raise ValueError("Invalid template file format")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load templates from {filepath}: {e}")
            return self._load_builtin_templates()

    def select_template(self, attributes: Dict[str, Any]) -> str:
        """
        Select a template based on attributes.

        Args:
            attributes: Decomposed concept attributes

        Returns:
            Selected template string
        """
        # Simple selection: cycle through templates for variety
        import random

        return random.choice(self.templates)

    def format_template(
        self, template: str, attributes: Dict[str, Any], concept: str
    ) -> str:
        """
        Format a template with attributes.

        Args:
            template: Template string
            attributes: Attributes for formatting
            concept: Original concept for {concept} placeholder

        Returns:
            Formatted string
        """
        # Create a copy to avoid modifying original
        attrs = dict(attributes)
        attrs["concept"] = concept

        # Safe formatting with fallback for missing keys
        try:
            return template.format(**attrs)
        except KeyError as e:
            logger.warning(f"Missing key {e} in template. Using fallback.")
            # Fallback template
            fallback = "{emotion} is like {element}"
            try:
                return fallback.format(**attrs)
            except KeyError:
                return attrs.get("emotion", "the feeling")


class SafetyFilter:
    """Handles content safety and filtering."""

    def __init__(self):
        self.forbidden_words = {"harm", "violence", "blood", "kill", "death", "suicide"}
        self.profanity_filter = None

        if PROFANITY_AVAILABLE:
            try:
                self.profanity_filter = ProfanityFilter()
                logger.info("Profanity filter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize profanity filter: {e}")

    def is_safe(self, text: str) -> bool:
        """
        Check if text is safe for output.

        Args:
            text: Text to check

        Returns:
            True if safe, False otherwise
        """
        if not text:
            return False

        # Basic keyword filtering
        text_lower = text.lower()
        if any(word in text_lower for word in self.forbidden_words):
            logger.warning(f"Text contains forbidden word: {text}")
            return False

        # Profanity filtering (if available)
        if self.profanity_filter:
            try:
                if self.profanity_filter.is_profane(text):
                    logger.warning(f"Text contains profanity: {text}")
                    return False
            except Exception as e:
                logger.warning(f"Profanity check failed: {e}")

        # Additional safety: Check for excessive repetition or patterns
        # (simple heuristic for potential spam/malicious content)
        if len(text) > 500:  # Unusually long
            logger.warning("Text is unusually long")
            return False

        return True


class ImageryEnhancer:
    """Enhances metaphors with visual details."""

    def __init__(self):
        self.imagery_templates = [
            ", visualized as {color} {shape} against {background}",
            ", conjuring {material} textures that {sensation} the skin",
            ", shimmering like {light_source} in {atmosphere}",
            ", with {sound} echoing through {space}",
        ]

        # Color/texture/visual database
        self.visual_library = {
            "color": [
                "soft amber",
                "deep indigo",
                "pale gold",
                "emerald",
                "crimson",
                "silver",
            ],
            "shape": [
                "swirling clouds",
                "fragmented mirrors",
                "liquid mercury",
                "crystalline shards",
            ],
            "background": [
                "velvet darkness",
                "dawn horizon",
                "infinite sky",
                "ocean depths",
            ],
            "material": ["silk", "crystal", "water", "smoke"],
            "sensation": ["brushes", "caresses", "dances across", "echoes through"],
            "light_source": ["starlight", "moonbeams", "sunfire", "bioluminescence"],
            "atmosphere": [
                "silent chambers",
                "open fields",
                "ancient ruins",
                "dreamscapes",
            ],
            "sound": ["whispers", "melodies", "thunder", "silence"],
            "space": ["the vastness", "empty halls", "forever night", "the heart"],
        }

    def enhance(self, metaphor: str) -> str:
        """
        Add visual imagery to a metaphor.

        Args:
            metaphor: Base metaphor string

        Returns:
            Enhanced metaphor with visual details
        """
        import random

        # Select imagery template
        template = random.choice(self.imagery_templates)

        # Fill template with random visual elements
        try:
            # Extract placeholders and fill with random choices
            filled_template = template
            for category, options in self.visual_library.items():
                if f"{{{category}}}" in template:
                    filled_template = filled_template.replace(
                        f"{{{category}}}", random.choice(options)
                    )

            return metaphor + filled_template
        except Exception as e:
            logger.warning(f"Failed to enhance imagery: {e}")
            return metaphor


class LLMGenerator:
    """Placeholder for LLM-based generation (requires transformers)."""

    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers not available. Install with: pip install transformers torch"
            )

        # Initialize a small model for generation (GPT-2 small)
        # In production, consider larger models or API-based solutions
        try:
            self.pipeline = pipeline(
                "text-generation",
                model="gpt2",
                max_length=100,
                num_return_sequences=3,
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("LLM generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def generate(self, concept: str, num_metaphors: int = 3) -> List[str]:
        """
        Generate metaphors using LLM.

        Args:
            concept: Input concept
            num_metaphors: Number of metaphors to generate

        Returns:
            List of generated metaphors
        """
        try:
            # Craft prompt for metaphor generation
            prompt = f"Generate a beautiful, poetic metaphor for the emotion/concept of '{concept}':"

            # Generate responses
            results = self.pipeline(
                prompt,
                max_length=150,
                num_return_sequences=num_metaphors,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            # Extract generated text and clean
            metaphors = []
            for result in results:
                generated = result["generated_text"]
                # Extract metaphor part (remove prompt)
                if prompt in generated:
                    metaphor_part = generated.replace(prompt, "").strip()
                    # Clean up to first sentence or reasonable length
                    metaphor_part = metaphor_part.split(".")[0]
                    if metaphor_part:
                        metaphors.append(metaphor_part)

            # If we got no metaphors, fall back to empty list
            if not metaphors:
                return []

            # Limit to requested number
            return metaphors[:num_metaphors]

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []


class MetaphorGenerator:
    """
    Main class for generating metaphors from concepts.

    Usage:
        generator = MetaphorGenerator(model_type='template')
        metaphors = generator.generate("grief", num_metaphors=3)
        print(metaphors)
    """

    def __init__(
        self,
        model_type: str = "template",
        templates_file: Optional[str] = None,
        use_nltk: bool = False,
    ):
        """
        Initialize the metaphor generator.

        Args:
            model_type: 'template' for rule-based, 'llm' for transformer-based
            templates_file: Optional path to custom templates JSON file
            use_nltk: Whether to use NLTK for enhanced decomposition
        """
        self.model_type = model_type
        self.concept = ""

        # Initialize components
        self.template_engine = TemplateEngine()
        self.safety_filter = SafetyFilter()
        self.imagery_enhancer = ImageryEnhancer()
        self.llm_generator = None

        # Load custom templates if provided
        if templates_file:
            self.templates = self.template_engine.load_from_file(templates_file)
        else:
            self.templates = self.template_engine.templates

        # Select decomposer based on NLTK preference
        if use_nltk:
            try:
                self.decomposer = NLTKDecomposer()
                logger.info("Using NLTK decomposer")
            except ImportError as e:
                logger.warning(
                    f"NLTK not available, falling back to keyword decomposer: {e}"
                )
                self.decomposer = KeywordDecomposer()
        else:
            self.decomposer = KeywordDecomposer()

        # Initialize LLM if requested
        if model_type == "llm":
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.llm_generator = LLMGenerator()
                    logger.info("LLM mode enabled")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM: {e}")
                    logger.info("Falling back to template mode")
                    self.model_type = "template"
            else:
                logger.warning(
                    "LLM mode requested but transformers not available. Using template mode."
                )
                self.model_type = "template"

        logger.info(
            f"MetaphorGenerator initialized with model_type='{self.model_type}'"
        )

    def _generate_template_metaphors(
        self, concept: str, num_metaphors: int
    ) -> List[str]:
        """
        Generate metaphors using template-based approach.

        Args:
            concept: Input concept
            num_metaphors: Number of metaphors to generate

        Returns:
            List of generated metaphors
        """
        metaphors = []
        attempts = 0
        max_attempts = num_metaphors * 3  # Allow extra attempts for safety filtering

        while len(metaphors) < num_metaphors and attempts < max_attempts:
            attempts += 1

            try:
                # Decompose concept
                attributes = self.decomposer.decompose(concept)

                # Select and format template
                template = self.template_engine.select_template(attributes)
                metaphor = self.template_engine.format_template(
                    template, attributes, concept
                )

                # Apply safety filter
                if not self.safety_filter.is_safe(metaphor):
                    continue

                # Enhance with imagery
                enhanced = self.imagery_enhancer.enhance(metaphor)

                # Ensure uniqueness
                if enhanced not in metaphors:
                    metaphors.append(enhanced)

            except Exception as e:
                logger.warning(f"Error generating metaphor attempt {attempts}: {e}")
                continue

        # If we couldn't generate enough metaphors, log a warning
        if len(metaphors) < num_metaphors:
            logger.warning(
                f"Only generated {len(metaphors)} out of {num_metaphors} metaphors for '{concept}'"
            )

        return metaphors

    def _generate_llm_metaphors(self, concept: str, num_metaphors: int) -> List[str]:
        """
        Generate metaphors using LLM approach.

        Args:
            concept: Input concept
            num_metaphors: Number of metaphors to generate

        Returns:
            List of generated metaphors
        """
        if not self.llm_generator:
            return []

        try:
            metaphors = self.llm_generator.generate(concept, num_metaphors)
            # Filter and enhance LLM outputs
            safe_metaphors = [m for m in metaphors if self.safety_filter.is_safe(m)]
            # Enhance with imagery
            enhanced = [self.imagery_enhancer.enhance(m) for m in safe_metaphors]
            return enhanced[:num_metaphors]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []

    def generate(self, concept: str, num_metaphors: int = 3) -> List[str]:
        """
        Generate metaphors for the given concept.

        Args:
            concept: Emotional or abstract concept (e.g., "grief", "joy")
            num_metaphors: Number of metaphors to generate (1-10)

        Returns:
            List of generated metaphor strings

        Raises:
            ValueError: If concept is invalid or num_metaphors out of range
        """
        # Validate inputs
        if not concept or not concept.strip():
            raise ValueError("Concept cannot be empty or whitespace")

        if (
            not isinstance(num_metaphors, int)
            or num_metaphors < 1
            or num_metaphors > 10
        ):
            raise ValueError("num_metaphors must be between 1 and 10")

        # Store concept for use in templates
        self.concept = concept

        # Use LLM if available and requested
        if self.model_type == "llm" and self.llm_generator:
            metaphors = self._generate_llm_metaphors(concept, num_metaphors)
            if metaphors:
                return metaphors

        # Fall back to template-based generation
        return self._generate_template_metaphors(concept, num_metaphors)

    def generate_single(self, concept: str) -> str:
        """
        Generate a single metaphor (convenience method).

        Args:
            concept: Emotional or abstract concept

        Returns:
            A single metaphor string
        """
        metaphors = self.generate(concept, num_metaphors=1)
        return metaphors[0] if metaphors else "No metaphor generated"

    def __repr__(self) -> str:
        return f"MetaphorGenerator(model_type='{self.model_type}')"


# Helper functions at module level
def filter_safe(output: str) -> bool:
    """
    Convenience function to check if output is safe.

    Args:
        output: String to check

    Returns:
        True if safe, False otherwise
    """
    filter_obj = SafetyFilter()
    return filter_obj.is_safe(output)


def load_templates() -> List[str]:
    """
    Load built-in templates (for backward compatibility).

    Returns:
        List of template strings
    """
    engine = TemplateEngine()
    return engine.templates


def load_imagery_enhancer() -> ImageryEnhancer:
    """
    Create an ImageryEnhancer instance.

    Returns:
        ImageryEnhancer instance
    """
    return ImageryEnhancer()


# Example usage and testing block
if __name__ == "__main__":
    print("Metaphor Generation Module - Test Run")
    print("=" * 50)

    # Example 1: Basic template generation
    print("\n1. Template-based generation (default):")
    generator = MetaphorGenerator(model_type="template")
    test_concepts = ["grief", "joy", "loneliness", "anxiety", "time"]

    for concept in test_concepts:
        try:
            metaphors = generator.generate(concept, num_metaphors=2)
            print(f"\nConcept: {concept}")
            for i, metaphor in enumerate(metaphors, 1):
                print(f"  {i}. {metaphor}")
        except Exception as e:
            print(f"  Error: {e}")

    # Example 2: Single metaphor convenience method
    print("\n" + "=" * 50)
    print("\n2. Single metaphor generation:")
    single = generator.generate_single("hope")
    print(f"  {single}")

    # Example 3: With NLTK (if available)
    if NLTK_AVAILABLE:
        print("\n" + "=" * 50)
        print("\n3. With NLTK decomposer:")
        nltk_gen = MetaphorGenerator(model_type="template", use_nltk=True)
        nltk_metaphors = nltk_gen.generate("fear", num_metaphors=2)
        for i, metaphor in enumerate(nltk_metaphors, 1):
            print(f"  {i}. {metaphor}")

    # Example 4: LLM mode (if transformers available)
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 50)
        print("\n4. LLM-based generation (if transformers installed):")
        llm_gen = MetaphorGenerator(model_type="llm")
        try:
            llm_metaphors = llm_gen.generate("love", num_metaphors=2)
            for i, metaphor in enumerate(llm_metaphors, 1):
                print(f"  {i}. {metaphor}")
        except Exception as e:
            print(f"  LLM generation failed: {e}")
    else:
        print("\n" + "=" * 50)
        print("\n4. LLM mode not available (transformers not installed)")
        print("   To enable: pip install transformers torch")

    # Example 5: Safety filtering test
    print("\n" + "=" * 50)
    print("\n5. Safety filter test:")
    safe_text = "A beautiful metaphor about nature"
    unsafe_text = "This is about harm and violence"
    print(f"  Safe: '{safe_text}' -> {filter_safe(safe_text)}")
    print(f"  Unsafe: '{unsafe_text}' -> {filter_safe(unsafe_text)}")

    # Example 6: Error handling
    print("\n" + "=" * 50)
    print("\n6. Error handling test:")
    try:
        generator.generate("", num_metaphors=3)
    except ValueError as e:
        print(f"  Empty concept error: {e}")

    try:
        generator.generate("test", num_metaphors=20)
    except ValueError as e:
        print(f"  Invalid num_metaphors error: {e}")

    print("\n" + "=" * 50)
    print("\nTest complete!")
