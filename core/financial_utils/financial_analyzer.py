import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from core.llm_api import run_llm

class BaseScorer(ABC):
    """Abstract base class for all financial scoring modules."""
    @abstractmethod
    async def score(self, data):
        pass

class LLMEnhancedScorer(BaseScorer):
    """A scorer that uses an LLM to evaluate financial data."""
    evaluation_prompt = ""

    async def score(self, data):
        if not self.evaluation_prompt:
            raise ValueError("Evaluation prompt cannot be empty for LLMEnhancedScorer.")

        prompt = f"""
        {self.evaluation_prompt}

        Analyze the following financial data. Based *only* on the data provided, provide a score from 1 to 100, where 1 is low and 100 is high. Your response must be a single integer.

        Financial Data:
        ---
        {str(data)[:4000]}
        ---

        Score (1-100):
        """

        try:
            response_dict = await run_llm(prompt, purpose="scoring", force_model=None)
            response = response_dict.get("result", "")
            score_str = ''.join(filter(str.isdigit, response))
            if score_str:
                score = int(score_str)
                return score / 100.0
            else:
                return 0.0
        except Exception as e:
            print(f"Error during LLM-enhanced financial scoring: {e}")
            return 0.0

class CryptoOpportunityScorer(LLMEnhancedScorer):
    """Rates the potential of a cryptocurrency based on market data."""
    evaluation_prompt = "Assess the investment potential of the following cryptocurrency data. Consider factors like price trends, market sentiment, and technological innovation."

class FinancialAnalyzer:
    """
    Analyzes financial data using a plugin architecture for different scoring modules.
    """
    def __init__(self, scorers):
        self.scorers = scorers

    async def analyze(self, data):
        """
        Runs all configured scorers on a set of financial data.
        """
        scores = {}
        for name, scorer_instance in self.scorers.items():
            score = await scorer_instance.score(data)
            scores[name] = round(score, 3)
        return scores
