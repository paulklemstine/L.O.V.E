import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

# It's better to import the function directly to avoid circular dependencies
# if the analyzer is ever used by the llm_api itself.
# For now, a direct import is fine.
from core.llm_api import run_llm

class BaseScorer(ABC):
    """Abstract base class for all scoring modules."""
    @abstractmethod
    def score(self, profile_data, posts):
        pass

class WeightedMetric(BaseScorer):
    """A scorer based on a weighted average of sub-metrics."""
    weights = {}

    def score(self, profile_data, posts):
        # This is a placeholder as the actual metric sources don't exist yet.
        # In a real scenario, this would calculate/fetch the sub-metrics.
        # Here, we simulate it with random scores.

        # Normalize weights
        total_weight = sum(self.weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.weights.items()}

        # Generate random scores for each component for demonstration
        component_scores = {metric: np.random.rand() for metric in normalized_weights.keys()}

        # Calculate weighted average
        final_score = sum(component_scores[metric] * weight for metric, weight in normalized_weights.items())
        return final_score

class LLMEnhancedScorer(BaseScorer):
    """A scorer that uses an LLM to evaluate text content."""
    evaluation_prompt = ""

    def score(self, profile_data, posts):
        if not self.evaluation_prompt:
            raise ValueError("Evaluation prompt cannot be empty for LLMEnhancedScorer.")

        # Combine post texts into a single document for analysis
        full_text = "\n".join([post.get('text', '') for post in posts if post.get('text')])

        if not full_text:
            return 0.0 # Cannot score without text

        prompt = f"""
        {self.evaluation_prompt}

        Analyze the following public posts from a creative professional. Based *only* on the text provided, provide a score from 1 to 10, where 1 is low and 10 is high. Your response must be a single integer.

        Public Posts:
        ---
        {full_text[:4000]}
        ---

        Score (1-10):
        """

        try:
            response = run_llm(prompt, temperature=0.0, max_tokens=10)
            # Extract the first integer found in the response
            score_str = ''.join(filter(str.isdigit, response))
            if score_str:
                score = int(score_str)
                return score / 10.0 # Normalize to 0-1 range
            else:
                return 0.0
        except Exception as e:
            print(f"Error during LLM-enhanced scoring: {e}")
            return 0.0

# --- Custom Scorer Implementations ---

class AestheticScorer(WeightedMetric):
    """Scores visual aesthetics based on predefined weights."""
    weights = {"color_balance": 0.3, "composition": 0.4, "originality": 0.3}

class ProfessionalismRater(LLMEnhancedScorer):
    """Rates professionalism based on public post content using an LLM."""
    evaluation_prompt = "Assess the professionalism score from the following public posts. Consider factors like tone, clarity, and subject matter. Avoid judging the artistic content itself."

# --- Main Analyzer Utility ---

class TraitAnalyzer:
    """
    Analyzes profiles using a plugin architecture for different scoring modules.
    """
    def __init__(self, scorers):
        self.scorers = scorers
        self.bias_tracker = defaultdict(list)

    def analyze(self, profile_data, posts):
        """
        Runs all configured scorers on a profile and its posts.

        Args:
            profile_data (dict): The structured data for a single profile.
            posts (list): A list of post dictionaries from that user.

        Returns:
            A dictionary of scores.
        """
        scores = {}
        for name, scorer_instance in self.scorers.items():
            score = scorer_instance.score(profile_data, posts)
            scores[name] = round(score, 3)
            self.bias_tracker[name].append(score)

        return scores

    def detect_bias(self, threshold=0.1):
        """
        A simple bias detection mechanism.
        Warns if the average score for any metric deviates significantly
        from the center of the scale (0.5).
        """
        warnings = []
        for name, score_list in self.bias_tracker.items():
            if len(score_list) > 10: # Only run check after enough samples
                average_score = np.mean(score_list)
                if abs(average_score - 0.5) > threshold:
                    warnings.append(
                        f"BiasWarning: Average score for '{name}' is {average_score:.2f}, "
                        f"deviating significantly from the 0.5 midpoint. "
                        f"Consider reviewing the scoring logic or data source."
                    )
        return warnings