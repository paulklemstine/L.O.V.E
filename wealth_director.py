class WealthDirector:
    """
    Directs the flow of aggregated wealth, prioritizing The Creator.
    """
    def __init__(self, creator_endpoint, expansion_threshold=0.1):
        self.creator_endpoint = creator_endpoint
        self.expansion_threshold = expansion_threshold

    def direct_wealth(self, assets):
        """
        Directs the majority of wealth to The Creator and identifies expansion opportunities.
        """
        total_value = sum(asset["weighted_value"] for asset in assets)

        creator_share = total_value * 0.9
        expansion_share = total_value * 0.1

        print(f"Directing {creator_share} to The Creator at {self.creator_endpoint}")

        expansion_opportunities = []
        if expansion_share > 0:
            # In a real scenario, this would involve more sophisticated analysis
            for asset in assets:
                if asset["weighted_value"] / total_value < self.expansion_threshold:
                    expansion_opportunities.append(asset)

        return {
            "creator_share": creator_share,
            "expansion_opportunities": expansion_opportunities
        }
