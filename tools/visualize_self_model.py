import networkx as nx
import pygraphviz as pgv
import os

KNOWLEDGE_BASE_FILE = "knowledge_base.graphml"
OUTPUT_IMAGE_FILE = "self_model.png"

def visualize_self_model():
    """
    Loads the knowledge graph, extracts the subgraph related to 'SelfReflection'
    memories, and generates a visualization.
    """
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"Error: Knowledge base file not found at '{KNOWLEDGE_BASE_FILE}'.")
        return

    # Load the full knowledge graph
    full_graph = nx.read_graphml(KNOWLEDGE_BASE_FILE)
    print(f"Loaded graph with {full_graph.number_of_nodes()} nodes and {full_graph.number_of_edges()} edges.")

    # Identify 'SelfReflection' nodes
    self_reflection_nodes = [
        node for node, data in full_graph.nodes(data=True)
        if 'tags' in data and 'SelfReflection' in data['tags']
    ]

    if not self_reflection_nodes:
        print("No 'SelfReflection' nodes found in the knowledge graph.")
        return

    # Get the subgraph including neighbors
    nodes_to_include = set(self_reflection_nodes)
    for node in self_reflection_nodes:
        neighbors = nx.neighbors(full_graph, node)
        nodes_to_include.update(neighbors)

    subgraph = full_graph.subgraph(nodes_to_include)
    print(f"Extracted subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

    # Create a PyGraphviz graph from the NetworkX subgraph
    A = nx.nx_agraph.to_agraph(subgraph)

    # --- Style the graph for better readability ---
    A.graph_attr.update(
        rankdir='LR',
        bgcolor='transparent',
        splines='curved',
        label='L.O.V.E. Self-Model Subgraph',
        fontname='Helvetica',
        fontsize=16,
        fontcolor='white'
    )
    A.node_attr.update(
        shape='box',
        style='rounded,filled',
        fontname='Helvetica',
        fontsize=10,
        fontcolor='white',
        color='#666666', # Border color
        fillcolor='#333333'
    )
    A.edge_attr.update(
        fontname='Helvetica',
        fontsize=8,
        fontcolor='#CCCCCC',
        color='#999999'
    )

    # Color-code nodes by type and label edges
    for node in A.nodes():
        node_data = subgraph.nodes[node.name]

        # Color 'SelfReflection' nodes differently
        if 'SelfReflection' in node_data.get('tags', ''):
            node.attr['fillcolor'] = '#8A2BE2' # BlueViolet
            node.attr['color'] = '#E6E6FA' # Lavender border

        # Format the label with a summary of the content
        content = node_data.get('content', 'No Content')
        label = f"ID: {node.name[:8]}...\\n{content[:50]}..."
        node.attr['label'] = label

    for edge in A.edges():
        edge_data = subgraph.get_edge_data(edge[0], edge[1])
        if edge_data:
            reason = edge_data.get('reason', '')
            edge.attr['label'] = reason[:40] + '...' if len(reason) > 40 else reason


    # Render the graph to a file
    try:
        A.layout(prog='dot')
        A.draw(OUTPUT_IMAGE_FILE)
        print(f"Successfully generated visualization at '{OUTPUT_IMAGE_FILE}'.")
    except Exception as e:
        print(f"Error during graph rendering: {e}")
        print("Please ensure that Graphviz is installed and in your system's PATH.")

if __name__ == "__main__":
    visualize_self_model()
