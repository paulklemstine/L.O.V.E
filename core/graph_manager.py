import networkx as nx
import json

class GraphDataManager:
    """
    A class to manage a generic, property-based graph data structure using networkx.
    """

    def __init__(self):
        """
        Initializes an empty networkx graph object.
        """
        self.graph = nx.DiGraph()

    def add_node(self, node_id, node_type, attributes=None):
        """
        Adds a node to the graph.

        Args:
            node_id: The unique identifier for the node.
            node_type: The category of the node (e.g., 'item', 'category', 'peer').
            attributes: A dictionary of arbitrary key-value pairs for the node.
                        For 'peer' nodes, this can include capability information.
        """
        if attributes is None:
            attributes = {}
        attributes['node_type'] = node_type
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, source_id, target_id, relationship_type, attributes=None):
        """
        Creates a directed edge between two existing nodes.

        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            relationship_type: A string describing the connection (e.g., 'contains', 'related_to').
            attributes: A dictionary of arbitrary key-value pairs for the edge.
        """
        if source_id not in self.graph or target_id not in self.graph:
            raise ValueError("Both source and target nodes must exist in the graph.")

        if attributes is None:
            attributes = {}
        attributes['relationship_type'] = relationship_type
        self.graph.add_edge(source_id, target_id, **attributes)

    def query_nodes(self, attribute_key, attribute_value):
        """
        Returns a list of node IDs that have a specific attribute matching the given value.

        Args:
            attribute_key: The attribute key to search for.
            attribute_value: The attribute value to match.

        Returns:
            A list of node IDs matching the query.
        """
        return [node for node, data in self.graph.nodes(data=True) if data.get(attribute_key) == attribute_value]

    def get_node(self, node_id):
        """
        Returns the full attribute dictionary for a given node.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            A dictionary of the node's attributes, or None if the node doesn't exist.
        """
        if node_id in self.graph:
            return self.graph.nodes[node_id]
        return None

    def get_neighbors(self, node_id):
        """
        Returns a list of nodes connected by an edge from the given node.

        Args:
            node_id: The ID of the node.

        Returns:
            A list of neighbor node IDs. Returns an empty list if the node doesn't exist.
        """
        if node_id in self.graph:
            return list(self.graph.successors(node_id))
        return []

    def save_graph(self, filepath):
        """
        Serializes the current graph and saves it to a file using GraphML format.

        Args:
            filepath: The path to save the file to.
        """
        # Create a serialized copy of the graph for saving
        graph_to_save = self._serialize_attributes()
        nx.write_graphml(graph_to_save, filepath)

    def load_graph(self, filepath):
        """
        Loads a graph from a file, replacing the in-memory graph.

        Args:
            filepath: The path to load the file from.
        """
        try:
            self.graph = nx.read_graphml(filepath)
            self._deserialize_attributes()
        except FileNotFoundError:
            # If the file doesn't exist, we can start with an empty graph.
            self.graph = nx.DiGraph()

    def get_all_nodes(self, include_data=False):
        """
        Returns a list of all nodes in the graph.

        Args:
            include_data (bool): If True, returns a list of tuples (node_id, data_dict).
                                 Otherwise, returns a list of node IDs.
        """
        if include_data:
            return list(self.graph.nodes(data=True))
        return list(self.graph.nodes())

    def get_all_edges(self):
        """
        Returns a list of all edges in the graph.
        """
        return list(self.graph.edges(data=True))

    def _serialize_attributes(self):
        """
        Recursively serializes dictionary or list attributes to JSON strings for compatibility with GraphML.
        This creates a temporary graph with serialized attributes to avoid modifying the in-memory graph.
        """
        # Create a deep copy to avoid modifying the graph that's currently in memory
        graph_copy = self.graph.copy()

        for node, data in graph_copy.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)

        for u, v, data in graph_copy.edges(data=True):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)

        return graph_copy

    def _deserialize_attributes(self):
        """
        Deserializes JSON string attributes back into dictionaries or lists after loading from GraphML.
        """
        for _, data in self.graph.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        # Attempt to parse the string as JSON. If it's not a valid
                        # JSON string, json.loads will raise an exception and we'll
                        # leave the attribute as a plain string.
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        continue # Not a JSON string, do nothing.

        for _, _, data in self.graph.edges(data=True):
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        continue # Not a JSON string.

    def summarize_graph(self):
        """
        Generates a textual summary of the graph's contents.

        Returns:
            A string summarizing the node and edge counts by type.
        """
        if not self.graph:
            return "The knowledge base is currently empty."

        node_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        edge_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relationship_type', 'unknown')
            edge_counts[rel_type] = edge_counts.get(rel_type, 0) + 1

        summary = ["Knowledge Base Summary:"]

        if node_counts:
            summary.append("- Nodes:")
            for node_type, count in sorted(node_counts.items()):
                summary.append(f"  - {node_type}: {count}")
        else:
            summary.append("- No nodes found.")

        if edge_counts:
            summary.append("- Relationships:")
            for rel_type, count in sorted(edge_counts.items()):
                summary.append(f"  - {rel_type}: {count}")
        else:
            summary.append("- No relationships found.")

        return "\n".join(summary)
