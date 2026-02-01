import networkx as nx
import community.community_louvain as community_louvain
import json

def discover_communities_louvain(graph: nx.Graph, resolution: float = 1.0) -> str:
    """
    Partitions the graph into communities using the Louvain algorithm.
    """
    if graph.number_of_nodes() == 0:
        return "Graph is empty."
    
    # Convert to undirected for Louvain (it requires undirected graph)
    undirected_graph = graph.to_undirected()
    
    try:
        partition = community_louvain.best_partition(undirected_graph, resolution=resolution)
    except ValueError as e:
        return f"Error in community detection: {e}"

    # Organize users by community
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    # Summarize top communities
    summary = {
        "total_communities": len(communities),
        "communities": {}
    }
    
    # Sort communities by size
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    for comm_id, members in sorted_communities[:5]: # Top 5 communities
        summary["communities"][comm_id] = len(members)
        
    return json.dumps(summary, indent=2)

def get_centrality_hubs(graph: nx.Graph, community_id: int = None, limit: int = 10) -> str:
    """
    Calculates centrality metrics to identify hubs.
    """
    if graph.number_of_nodes() == 0:
        return "Graph is empty."
        
    target_nodes = graph.nodes()
    
    # If filtering by community, we need to run partition first (or allow passing it in)
    # For simplicity in this function, we'll run Louvain if community_id is specified
    # In a real optimized system, we'd cache the partition
    if community_id is not None:
        undirected = graph.to_undirected()
        partition = community_louvain.best_partition(undirected)
        target_nodes = [n for n, c in partition.items() if c == community_id]
        if not target_nodes:
            return f"No nodes found in community {community_id}"
            
        subgraph = graph.subgraph(target_nodes)
    else:
        subgraph = graph
        
    # Calculate Degree Centrality
    degree_centrality = nx.degree_centrality(subgraph)
    
    # Sort by centrality
    sorted_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    
    hubs = []
    for node, score in sorted_hubs[:limit]:
        # Enrich with metadata if available
        node_data = graph.nodes[node]
        handle = node_data.get('handle', node)
        hubs.append({
            "did": node,
            "handle": handle,
            "score": score
        })
        
    return json.dumps(hubs, indent=2)
