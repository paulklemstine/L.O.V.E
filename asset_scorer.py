from market_data_harvester import get_crypto_market_data, get_nft_collection_stats
from core.llm_api import run_llm
import json
import asyncio

class AssetScorer:
    def __init__(self, weights):
        self.weights = weights

    async def calculate_score(self, asset):
        """
        Calculates a weighted score for a given asset based on multiple criteria.
        This method is now asynchronous to support LLM calls.
        It returns both the final score and the detailed breakdown of individual scores.
        """
        score = 0
        # Use asyncio.gather to run scoring functions concurrently
        scoring_results = await asyncio.gather(
            self._passive_income_potential_score(asset),
            self._aesthetic_alignment_score(asset),
            # Add other scoring functions as needed
        )

        score_details = {
            "passive_income_potential": scoring_results[0],
            "aesthetic_alignment": scoring_results[1]
        }

        # Apply weights to the results
        score += score_details["passive_income_potential"] * self.weights.get("passive_income_potential", 0)
        score += score_details["aesthetic_alignment"] * self.weights.get("aesthetic_alignment", 0)

        return min(score, 100), score_details

    async def _passive_income_potential_score(self, asset):
        """
        Scores an asset based on its potential to generate passive income.
        """
        asset_type = asset.get('type')
        if asset_type == 'cryptocurrency':
            market_data = get_crypto_market_data([asset.get('id')])
            if not market_data:
                return 0
            # Example logic: Higher score for coins with high market cap and low volatility (to be implemented)
            # For now, we'll use a simple score based on market cap rank.
            market_cap_rank = market_data[0].get('market_cap_rank', 1000)
            return max(0, 100 - (market_cap_rank / 10)) # Higher rank (lower number) = higher score
        elif asset_type == 'nft_collection':
            stats = get_nft_collection_stats(asset.get('slug'))
            if not stats:
                return 0
            # Example logic: Higher score for collections with high trading volume and a low floor price.
            volume = stats.get('total_volume', 0)
            floor_price = stats.get('floor_price', 1)
            # Normalize and combine. This is a placeholder for a more sophisticated model.
            return min(100, (volume / 1000) * (1 / (floor_price + 0.1)))
        return 0

    async def _aesthetic_alignment_score(self, asset):
        """
        Uses an LLM to score an NFT collection's alignment with The Creator's aesthetic preferences.
        """
        if asset.get('type') != 'nft_collection':
            return 50 # Neutral score for non-NFT assets

        # In a real implementation, we would fetch more detailed metadata.
        # For now, we'll use the asset's name and slug as a proxy for its aesthetic.
        collection_name = asset.get('name', '')
        collection_slug = asset.get('slug', '')

        prompt = f"""
        Analyze the following NFT collection details for its aesthetic alignment with "Hawaii candy vibes + Neo-matrix".
        The aesthetic is bright, neon, candy-like colors against a dark background, with rave emojis and binary ASCII art.

        Collection Name: {collection_name}
        Collection Slug: {collection_slug}

        Based on this information, provide a score from 0 to 100 indicating the likelihood of aesthetic alignment.
        Respond with only a single integer.
        """

        response_dict = await run_llm(prompt, purpose="aesthetic_scoring")
        score_str = response_dict.get("result", "0").strip()

        try:
            return int(score_str)
        except (ValueError, TypeError):
            return 0 # Return a default score if parsing fails

def allocate_resources(value, creator_allocation_ratio=0.9):
    """
    Allocates the value of an acquired asset. 90% is directed to The Creator by default.
    """
    creator_share = value * creator_allocation_ratio
    self_preservation_share = value * (1 - creator_allocation_ratio)

    return {
        "creator_share": creator_share,
        "self_preservation_share": self_preservation_share,
    }
