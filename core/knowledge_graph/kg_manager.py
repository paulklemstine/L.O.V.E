import json

class KnowledgeGraphManager:
    def __init__(self):
        # In a real implementation, this would connect to a graph database.
        # For now, we just log the data to demonstrate the flow.
        self.graph_data = {"nodes": [], "edges": []}
        print("KnowledgeGraphManager: Initialized (simulation mode).")

    def update_graph(self, extracted_data):
        """
        Updates the knowledge graph with extracted entities and relationships.
        """
        entities = extracted_data.get("entities", [])
        relationships = extracted_data.get("relationships", [])

        for entity in entities:
            if entity not in self.graph_data["nodes"]:
                self.graph_data["nodes"].append(entity)

        for rel in relationships:
            edge = (rel["subject"], rel["predicate"], rel["object"])
            if edge not in self.graph_data["edges"]:
                self.graph_data["edges"].append(edge)

        print(f"KnowledgeGraphManager: Updated graph with {len(entities)} entities and {len(relationships)} relationships.")
        # In a real scenario, this would commit transactions to Neo4j, etc.
        self.log_graph_state()

    def log_graph_state(self):
        """Logs the current state of the simulated graph."""
        print("--- Knowledge Graph State (Simulated) ---")
        print(json.dumps(self.graph_data, indent=2))
        print("-----------------------------------------")