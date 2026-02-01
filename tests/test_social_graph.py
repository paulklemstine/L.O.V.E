import pytest
import networkx as nx
import json
import sys
import os

# Ensure we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcp_servers.social_graph.analysis import discover_communities_louvain, get_centrality_hubs

def test_community_detection():
    # Karate club graph is a standard benchmark for community detection
    G = nx.karate_club_graph()
    result = discover_communities_louvain(G)
    data = json.loads(result)
    
    assert "total_communities" in data
    assert data["total_communities"] > 1
    assert "communities" in data

def test_centrality():
    # Star graph: node 0 connected to 1,2,3...
    G = nx.star_graph(5)
    result = get_centrality_hubs(G, limit=1)
    data = json.loads(result)
    
    assert len(data) == 1
    # Node 0 is the center
    assert data[0]["did"] == 0
    
def test_empty_graph():
    G = nx.Graph()
    result = discover_communities_louvain(G)
    assert result == "Graph is empty."
    
    result = get_centrality_hubs(G)
    assert result == "Graph is empty."
