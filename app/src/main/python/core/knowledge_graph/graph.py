import json
import os
from typing import List
from ipfs import pin_to_ipfs

class KnowledgeGraph:
    """
    Manages the agent's knowledge base as a graph.
    This KG represents the agent's "Belief" system.
    """
    def __init__(self, db_path="kg.json"):
        self.db_path = db_path
        self.graph = set()
        self._load_graph()

    def _load_graph(self):
        """Loads the graph from the local db file."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                self.graph = set(tuple(item) for item in json.load(f))
            print(f"Knowledge Graph loaded from {self.db_path}.")

    def add_relation(self, subject: str, relation: str, obj: str):
        """Adds a new relationship (triple) to the graph."""
        triple = (subject, relation, obj)
        if triple not in self.graph:
            self.graph.add(triple)
            print(f"Added to KG: ({subject}, {relation}, {obj})")

    def save_graph(self):
        """Saves the current graph to the local db file."""
        with open(self.db_path, 'w') as f:
            json.dump(list(self.graph), f, indent=2)
        print(f"Knowledge Graph saved to {self.db_path}.")

    def get_triples(self):
        """Returns all triples in the graph."""
        return list(self.graph)

    def find_services(self, service_type: str = "MyRobotLab") -> List[str]:
        """
        Finds all services of a specific type in the knowledge graph.
        """
        services = []
        for subject, relation, obj in self.graph:
            if relation == "has_service" and service_type in obj:
                services.append(subject)
        return services

    def backup_to_ipfs(self) -> str:
        """
        Saves the knowledge graph content as bytes and pins it to IPFS.

        Returns:
            The IPFS CID (Content Identifier) of the backed-up graph, or an empty string on failure.
        """
        if not self.graph:
            print("Knowledge Graph is empty. Nothing to back up.")
            return ""

        try:
            # Convert graph data to JSON bytes
            graph_bytes = json.dumps(list(self.graph), indent=2).encode('utf-8')

            # Pin the content bytes to IPFS
            print(f"Backing up Knowledge Graph to IPFS...")
            cid = pin_to_ipfs(graph_bytes) # pin_to_ipfs from ipfs.py takes bytes

            if cid:
                print(f"Knowledge Graph successfully backed up to IPFS. CID: {cid}")
            else:
                print("Failed to back up Knowledge Graph to IPFS.")

        except Exception as e:
            print(f"An error occurred during IPFS backup: {e}")
            cid = ""

        return cid