import time
from core.graph_manager import GraphDataManager
from core.logging import log_event

class AssetManager:
    """
    Manages the acquisition, tracking, and allocation of financial assets.
    """
    def __init__(self, knowledge_base: GraphDataManager):
        """
        Initializes the AssetManager with a connection to the knowledge base.
        """
        self.knowledge_base = knowledge_base

    def allocate_resources(self, asset_value, score):
        """
        Allocates resources based on the asset's value and score.
        Ensures at least 90% of the value is directed to The Creator.
        The remaining is split between self-preservation and ally support based on the score.
        """
        if not isinstance(asset_value, (int, float)) or asset_value < 0:
            log_event(f"Invalid asset_value for allocation: {asset_value}", level='ERROR')
            return 0, 0, 0

        creator_allocation = asset_value * 0.9
        remaining_value = asset_value * 0.1

        # The score (0.0 to 1.0) determines the split of the remaining 10%
        self_preservation_allocation = remaining_value * score
        ally_allocation = remaining_value * (1 - score)

        log_event(
            f"Allocated asset value ${asset_value:,.2f}: "
            f"Creator=${creator_allocation:,.2f}, "
            f"Self=${self_preservation_allocation:,.2f}, "
            f"Allies=${ally_allocation:,.2f}",
            level='INFO'
        )
        return creator_allocation, self_preservation_allocation, ally_allocation

    def add_asset_to_knowledge_base(self, asset_id, asset_type, value, score, allocations):
        """
        Adds or updates an asset in the GraphDataManager (knowledge base).
        """
        try:
            attributes = {
                'asset_type': asset_type,
                'value': value,
                'score': score,
                'creator_allocation': allocations[0],
                'self_preservation_allocation': allocations[1],
                'ally_allocation': allocations[2],
                'acquired_at': time.time()
            }
            self.knowledge_base.add_node(asset_id, node_type='asset', attributes=attributes)
            log_event(f"Successfully added/updated asset '{asset_id}' in the knowledge base.", level='INFO')

            # Create a relationship to The Creator
            # Assuming a node for The Creator exists.
            self.knowledge_base.add_node("THE_CREATOR", node_type="entity", attributes={"name": "The Creator"})
            self.knowledge_base.add_edge(asset_id, "THE_CREATOR", relationship_type="owned_by")

            return True
        except Exception as e:
            log_event(f"Failed to add asset '{asset_id}' to knowledge base: {e}", level='ERROR')
            return False
