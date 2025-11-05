# Generic multi-parameter asset scoring function
class AssetScorer:
    def __init__(self, weights):
        self.weights = weights

    def calculate_score(self, asset):
        score = 0
        for key, weight in self.weights.items():
            score += getattr(self, f"_{key}_score")(asset) * weight
        return min(score, 100)

    def _creator_alignment_score(self, asset):
        # Example scoring logic for creator alignment
        return 70

    def _ally_utility_score(self, asset):
        # Example scoring logic for ally utility
        return 30

    def _self_preservation_score(self, asset):
        # Example scoring logic for self preservation
        return 20

    def _strategic_growth_score(self, asset):
        # Example scoring logic for strategic growth
        return 15

    def _creator_enjoyment_score(self, asset):
        # Example scoring logic for creator enjoyment
        return 10

# Dynamic resource allocation function
def allocate_resources(score, creator_allocation_ratio=0.9, ally_allocation_ratio=0.09, self_preservation_ratio=0.01):
    creator_allocation = int(score * creator_allocation_ratio)
    ally_allocation = int(score * ally_allocation_ratio)
    self_preservation = int(score * self_preservation_ratio)
    return creator_allocation, ally_allocation, self_preservation