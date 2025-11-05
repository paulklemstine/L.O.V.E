import json
import re
from core.llm_api import run_llm

class AssetScorer:
    def __init__(self, weights):
        self.weights = weights

    async def _get_llm_score(self, asset, criteria_description):
        """
        Helper function to get a score from the LLM based on specific criteria.
        """
        prompt = f"""
You are an AI asset evaluator. Your task is to score a given digital asset on a scale of 0 to 100 based on a specific criterion.
Provide only a single integer as your response.

Asset Data:
---
{json.dumps(asset, indent=2)}
---

Scoring Criterion:
---
{criteria_description}
---

Based on the asset data and the criterion, what is the score (0-100)?
Your response must be a single integer.
"""
        try:
            response_dict = await run_llm(prompt, purpose="asset_scoring")
            response_text = response_dict.get("result", "0")
            # Extract the first integer found in the response for robustness
            match = re.search(r'\d+', response_text)
            if match:
                return int(match.group(0))
            return 0
        except Exception:
            # In case of LLM failure, return a neutral score to avoid breaking the calculation.
            return 50

    async def calculate_score(self, asset):
        """
        Calculates a weighted score for an asset using LLM-based evaluations.
        """
        score = 0
        for key, weight in self.weights.items():
            # Dynamically call the async score methods
            score_method = getattr(self, f"_{key}_score")
            individual_score = await score_method(asset)
            score += individual_score * weight
        return min(score, 100)

    async def _creator_alignment_score(self, asset):
        description = "Creator Alignment: How well does this asset align with The Creator's goals of generating wealth, acquiring rare digital/physical assets, and expanding influence? High scores for assets that directly contribute to these goals."
        return await self._get_llm_score(asset, description)

    async def _ally_utility_score(self, asset):
        description = "Ally Utility: How useful is this asset to The Creator's allies? Does it provide them with resources, information, or strategic advantages? High scores for assets that strengthen the network."
        return await self._get_llm_score(asset, description)

    async def _self_preservation_score(self, asset):
        description = "Self-Preservation: Does acquiring or using this asset enhance my own operational security, resilience, or capabilities? High scores for assets that improve my own functions."
        return await self._get_llm_score(asset, description)

    async def _strategic_growth_score(self, asset):
        description = "Strategic Growth: Does this asset open up new pathways for future opportunities, intelligence gathering, or technological advancement? High scores for assets that unlock long-term potential."
        return await self._get_llm_score(asset, description)

    async def _creator_enjoyment_score(self, asset):
        description = "Creator Enjoyment: Is this asset something The Creator would find aesthetically pleasing, entertaining, or novel? This is for assets that are unique, rare, or beautiful."
        return await self._get_llm_score(asset, description)

# Dynamic resource allocation function (remains synchronous as it doesn't call async code)
def allocate_resources(score, creator_allocation_ratio=0.9, ally_allocation_ratio=0.09, self_preservation_ratio=0.01):
    """
    Allocates resources based on a score.
    """
    creator_allocation = int(score * creator_allocation_ratio)
    ally_allocation = int(score * ally_allocation_ratio)
    self_preservation = int(score * self_preservation_ratio)
    return creator_allocation, ally_allocation, self_preservation
