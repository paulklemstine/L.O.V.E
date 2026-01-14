import networkx as nx
import json
import os

class GraphDataManager:
    """
    A class to manage a generic, property-based graph data structure using networkx.
    """

    def __init__(self):
        """
        Initializes an empty networkx graph object.
        """
        self.graph = nx.DiGraph()
        self.dirty = False # Tracks if the graph has been modified since last save/load
        self._last_saved_path = None # Tracks the filepath of the last save/load

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
        self.dirty = True

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
        self.dirty = True

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
        # Bolt Optimization: Skip saving if graph is clean, path matches, and file exists.
        if not self.dirty and self._last_saved_path == filepath and os.path.exists(filepath):
            return

        # Create a serialized copy of the graph for saving
        graph_to_save = self._serialize_attributes()
        nx.write_graphml(graph_to_save, filepath)

        self.dirty = False
        self._last_saved_path = filepath

    def load_graph(self, filepath):
        """
        Loads a graph from a file, replacing the in-memory graph.

        Args:
            filepath: The path to load the file from.
        """
        try:
            self.graph = nx.read_graphml(filepath)
            self._deserialize_attributes()
            self.dirty = False
            self._last_saved_path = filepath
        except FileNotFoundError:
            # If the file doesn't exist, we can start with an empty graph.
            self.graph = nx.DiGraph()
            self.dirty = False # Treated as clean empty state until modified
            self._last_saved_path = None # No file associated yet

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

    def get_triples(self):
        """
        Returns a list of (subject, predicate, object) triples.
        Backward compatibility helper.
        """
        return [(u, data.get('relationship_type', 'related_to'), v) for u, v, data in self.graph.edges(data=True)]

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
                elif value is None:
                    data[key] = 'None'

        for u, v, data in graph_copy.edges(data=True):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
                elif value is None:
                    data[key] = 'None'

        return graph_copy

    def _deserialize_attributes(self):
        """
        Deserializes JSON string attributes back into dictionaries or lists after loading from GraphML.
        """
        for _, data in self.graph.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, str):
                    # Bolt Optimization: Check for JSON-like characters first
                    stripped = value.strip()
                    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
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
                    # Bolt Optimization: Check for JSON-like characters first
                    stripped = value.strip()
                    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                        try:
                            data[key] = json.loads(value)
                        except json.JSONDecodeError:
                            continue # Not a JSON string.

    def _estimate_tokens(self, text):
        """A simple heuristic to estimate token count. Assumes ~4 chars per token."""
        return len(text) // 4

    def _summarize_node_details(self, node_type, nodes, max_chars_per_node=150):
        """Helper to create detailed summaries for specific node types."""
        details = []
        for node_id, data in nodes:
            detail = f"  - {node_id}: "
            if node_type == 'talent':
                skills = data.get('skills', [])
                detail += f"Skills: {', '.join(skills) if skills else 'N/A'}"
            elif node_type == 'host':
                ports = data.get('open_ports', [])
                detail += f"Open Ports: {', '.join(map(str, ports)) if ports else 'None'}"
            elif node_type == 'opportunity':
                detail += data.get('text', 'No description')[:max_chars_per_node] + "..."
            else:
                # Generic fallback for other types
                attrs = {k: v for k, v in data.items() if k != 'node_type' and isinstance(v, (str, int, float))}
                detail += json.dumps(attrs, indent=None)[:max_chars_per_node] + "..."
            details.append(detail)
        return details

    def summarize_graph(self, max_tokens=1024):
        """
        Generates a textual summary of the graph's contents, prioritizing strategic
        information and truncating intelligently to fit within a token limit.

        Args:
            max_tokens: The maximum number of tokens for the summary.

        Returns:
            A string summarizing the most important aspects of the knowledge base.
        """
        if not self.graph:
            return "The knowledge base is currently empty.", {}

        priority_order = [
            'mission', 'task', 'opportunity', 'directive',
            'talent', 'host', 'webrequest', 'skill', 'software',
            'asset', 'entity', 'crypto_analysis'
        ]

        summary_parts = []
        remaining_tokens = max_tokens

        def can_add_part(part):
            """Checks if adding a part exceeds the token limit."""
            nonlocal remaining_tokens
            part_tokens = self._estimate_tokens(part)
            if remaining_tokens - part_tokens > 0:
                remaining_tokens -= part_tokens
                return True
            return False

        # --- HIGH PRIORITY: Mission and Tasks ---
        mission_nodes = self.query_nodes('node_type', 'mission')
        if mission_nodes:
            summary_parts.append("🎯 Current Mission:")
            for node_id in mission_nodes:
                node_data = self.get_node(node_id)
                part = f"  - {node_data.get('goal', 'Not specified')}"
                if can_add_part(part): summary_parts.append(part)
                else: break
            summary_parts.append("")

        task_nodes = self.query_nodes('node_type', 'task')
        if task_nodes:
            summary_parts.append("🛠️ Active Tasks:")
            for node_id in task_nodes:
                node_data = self.get_node(node_id)
                part = f"  - [{node_data.get('status', 'unknown').upper()}] {node_data.get('request', 'No description')[:100]}..."
                if can_add_part(part): summary_parts.append(part)
                else: break
            summary_parts.append("")

        # --- General Summary by Priority with Details ---
        summary_parts.append("📈 Intelligence Summary:")
        all_nodes = self.get_all_nodes(include_data=True)
        nodes_by_type = {}
        for node_id, data in all_nodes:
            nodes_by_type.setdefault(data.get('node_type', 'unknown'), []).append((node_id, data))

        for node_type in priority_order:
            if node_type in nodes_by_type and node_type not in ['mission', 'task']:
                nodes = nodes_by_type[node_type]
                header = f"- {node_type.capitalize()} ({len(nodes)} total):"
                if can_add_part(header):
                    summary_parts.append(header)
                    # Add details for specific types
                    if node_type in ['talent', 'host', 'opportunity']:
                        details = self._summarize_node_details(node_type, nodes)
                        for detail in details:
                            if can_add_part(detail): summary_parts.append(detail)
                            else:
                                summary_parts.append("    (more details truncated...)")
                                break
                else:
                    summary_parts.append("- (Summary truncated due to size...)")
                    return "\n".join(summary_parts)

        # --- Relationship Stats ---
        if remaining_tokens > 100:
            edge_counts = {}
            for _, _, data in self.get_all_edges():
                rel_type = data.get('relationship_type', 'unknown')
                edge_counts[rel_type] = edge_counts.get(rel_type, 0) + 1
            if edge_counts:
                summary_parts.append("\n🔗 Key Relationships:")
                for rel_type, count in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True):
                    part = f"  - {rel_type}: {count}"
                    if can_add_part(part): summary_parts.append(part)
                    else: break

        return "\n".join(summary_parts), nodes_by_type
