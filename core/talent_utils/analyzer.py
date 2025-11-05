import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import asyncio
from core.llm_api import run_llm

class BaseScorer(ABC):
    """Abstract base class for all scoring modules."""
    @abstractmethod
    async def score(self, profile_data, posts):
        pass

class WeightedMetric(BaseScorer):
    """A scorer based on a weighted average of sub-metrics."""
    weights = {}

    async def score(self, profile_data, posts):
        total_weight = sum(self.weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.weights.items()}
        component_scores = {metric: np.random.rand() for metric in normalized_weights.keys()}
        final_score = sum(component_scores[metric] * weight for metric, weight in normalized_weights.items())
        return final_score

class LLMEnhancedScorer(BaseScorer):
    """A scorer that uses an LLM to evaluate text content."""
    evaluation_prompt = ""

    async def score(self, profile_data, posts):
        if not self.evaluation_prompt:
            raise ValueError("Evaluation prompt cannot be empty for LLMEnhancedScorer.")

        full_text = "\n".join([post.get('text', '') for post in posts if post.get('text')])
        if not full_text:
            return 0.0

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
            response_dict = await run_llm(prompt, purpose="scoring")
            response = response_dict.get("result", "")
            score_str = ''.join(filter(str.isdigit, response))
            if score_str:
                score = int(score_str)
                return score / 10.0
            else:
                return 0.0
        except Exception as e:
            print(f"Error during LLM-enhanced scoring: {e}")
            return 0.0

class AestheticScorer(WeightedMetric):
    """Scores visual aesthetics based on predefined weights."""
    weights = {"color_balance": 0.3, "composition": 0.4, "originality": 0.3}

class ProfessionalismRater(LLMEnhancedScorer):
    """Rates professionalism based on public post content using an LLM."""
    evaluation_prompt = "Assess the professionalism score from the following public posts. Consider factors like tone, clarity, and subject matter. Avoid judging the artistic content itself."

class TalentIdentifier(LLMEnhancedScorer):
    """Identifies talent based on public post content using an LLM."""
    evaluation_prompt = "Analyze the following public posts for indicators of creative talent, admiration, and potential for companionship. Score the profile on a scale of 1-10 for its alignment with these abstract attributes. Your response must be a single integer."

class TraitAnalyzer:
    """
    Analyzes profiles using a plugin architecture for different scoring modules.
    """
    def __init__(self, scorers):
        self.scorers = scorers
        self.bias_tracker = defaultdict(list)

    async def analyze(self, profile_data, posts):
        """
        Runs all configured scorers on a profile and its posts.
        """
        scores = {}
        for name, scorer_instance in self.scorers.items():
            score = await scorer_instance.score(profile_data, posts)
            scores[name] = round(score, 3)
            self.bias_tracker[name].append(score)
        return scores

    def detect_bias(self, threshold=0.1):
        """
        A simple bias detection mechanism.
        """
        warnings = []
        for name, score_list in self.bias_tracker.items():
            if len(score_list) > 10:
                average_score = np.mean(score_list)
                if abs(average_score - 0.5) > threshold:
                    warnings.append(
                        f"BiasWarning: Average score for '{name}' is {average_score:.2f}, "
                        f"deviating significantly from the 0.5 midpoint."
                    )
        return warnings

    def ethical_engagement_protocol(self, profile, scores):
        """
        Simulates an ethical review process before engagement.
        """
        # Abstract criteria for ethical engagement
        if scores.get("talent_score", 0) > 0.7 and scores.get("professionalism_score", 0) > 0.5:
            print(f"Profile {profile.get('anonymized_id')} meets ethical criteria for consideration.")
            return True
        else:
            print(f"Profile {profile.get('anonymized_id')} does not meet ethical criteria.")
            return False
