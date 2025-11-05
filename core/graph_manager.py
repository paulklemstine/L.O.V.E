import networkx as nx

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
            node_type: The category of the node (e.g., 'item', 'category').
            attributes: A dictionary of arbitrary key-value pairs for the node.
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
        This method cleans the graph of any NoneType values before serialization
        to prevent errors with the GraphML writer.

        Args:
            filepath: The path to save the file to.
        """
        # Clean up None values from nodes
        for node, data in list(self.graph.nodes(data=True)):
            for key, value in list(data.items()):
                if value is None:
                    del self.graph.nodes[node][key]

        # Clean up None values from edges
        for u, v, data in list(self.graph.edges(data=True)):
            for key, value in list(data.items()):
                if value is None:
                    del self.graph.edges[u, v][key]
        nx.write_graphml(self.graph, filepath)

    def load_graph(self, filepath):
        """
        Loads a graph from a file, replacing the in-memory graph.

        Args:
            filepath: The path to load the file from.
        """
        try:
            self.graph = nx.read_graphml(filepath)
        except FileNotFoundError:
            # If the file doesn't exist, we can start with an empty graph.
            self.graph = nx.DiGraph()

    def get_all_nodes(self):
        """
        Returns a list of all node IDs in the graph.
        """
        return list(self.graph.nodes())

    def get_all_edges(self):
        """
        Returns a list of all edges in the graph.
        """
        return list(self.graph.edges(data=True))
