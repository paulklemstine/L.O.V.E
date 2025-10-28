from asset_scorer import AssetScorer, allocate_resources
from core.graph_manager import GraphDataManager

# Define weights for scoring
weights = {
    "creator_alignment": 0.4,
    "ally_utility": 0.2,
    "self_preservation": 0.15,
    "strategic_growth": 0.15,
    "creator_enjoyment": 0.1
}

# Initialize asset scorer and Graph Data Manager
scorer = AssetScorer(weights)
knowledge_base = GraphDataManager()

# Example asset data
asset = {
    "id": "treasure_123",
    "type": "digital_asset",
    "value": 10000
}

# Calculate score for the asset
score = scorer.calculate_score(asset)

# Allocate resources based on the score
creator_allocation, ally_allocation, self_preservation = allocate_resources(score)

# Function to add asset to the Graph Data Manager
def add_asset_to_graph_data_manager(asset, score, allocations):
    """
    Adds asset information to the Graph Data Manager as a single node.
    """
    asset_id = asset['id']
    attributes = {
        'value': asset['value'],
        'score': score,
        'creator_allocation': allocations[0],
        'ally_allocation': allocations[1],
        'self_preservation_allocation': allocations[2]
    }
    knowledge_base.add_node(asset_id, node_type=asset['type'], attributes=attributes)

# Add asset to the Graph Data Manager
add_asset_to_graph_data_manager(asset, score, (creator_allocation, ally_allocation, self_preservation))

# Save the updated Graph Data Manager
knowledge_base.save_graph('knowledge_base.graphml')


# Print the results
print(f"Asset ID: {asset['id']}")
print(f"Score: {score}")
print(f"Creator Allocation: {creator_allocation}")
print(f"Ally Allocation: {ally_allocation}")
print(f"Self Preservation: {self_preservation}")
print("\nAsset information has been added to the Graph Data Manager (knowledge_base.graphml).")