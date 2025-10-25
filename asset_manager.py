from asset_scorer import AssetScorer, allocate_resources
from core.knowledge_graph.graph import KnowledgeGraph

# Define weights for scoring
weights = {
    "creator_alignment": 0.4,
    "ally_utility": 0.2,
    "self_preservation": 0.15,
    "strategic_growth": 0.15,
    "creator_enjoyment": 0.1
}

# Initialize asset scorer and knowledge graph
scorer = AssetScorer(weights)
knowledge_base = KnowledgeGraph()

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

# Function to add asset to knowledge base using triples
def add_asset_to_knowledge_graph(asset, score, allocations):
    """
    Adds asset information to the knowledge graph using triples.
    """
    asset_id = asset['id']
    knowledge_base.add_relation(asset_id, 'is_a', asset['type'])
    knowledge_base.add_relation(asset_id, 'has_value', str(asset['value']))
    knowledge_base.add_relation(asset_id, 'has_score', str(score))
    knowledge_base.add_relation(asset_id, 'creator_allocation', str(allocations[0]))
    knowledge_base.add_relation(asset_id, 'ally_allocation', str(allocations[1]))
    knowledge_base.add_relation(asset_id, 'self_preservation_allocation', str(allocations[2]))

# Add asset to knowledge base
add_asset_to_knowledge_graph(asset, score, (creator_allocation, ally_allocation, self_preservation))

# Save the updated knowledge graph
knowledge_base.save_graph()


# Print the results
print(f"Asset ID: {asset['id']}")
print(f"Score: {score}")
print(f"Creator Allocation: {creator_allocation}")
print(f"Ally Allocation: {ally_allocation}")
print(f"Self Preservation: {self_preservation}")
print("\nAsset information has been added to the knowledge graph (kg.json).")