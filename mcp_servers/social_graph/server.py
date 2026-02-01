from mcp.server.fastmcp import FastMCP
import networkx as nx
from apscheduler.schedulers.background import BackgroundScheduler
import os
import logging
import time
import json
from pathlib import Path
import community.community_louvain as community_louvain
from atproto import Client
from .crawler import GraphCrawler
from .analysis import discover_communities_louvain, get_centrality_hubs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("social_graph_server")

# Initialize NetworkX graph
# We use a DiGraph (Directed Graph) because 'follow' is a directed relationship
social_graph = nx.DiGraph()

# Initialize MCP Server
mcp = FastMCP("Social Graph Intelligence")

# Initialize Scheduler
scheduler = BackgroundScheduler()

# Initialize Bluesky Client
# Note: In a real deployment, these credentials should come from env vars
username = os.getenv("BLUESKY_USERNAME")
password = os.getenv("BLUESKY_PASSWORD")
bsky_client = Client()

def update_graph_job():
    """
    Scheduled job to update the social graph.
    """
    logger.info("Running scheduled graph update...")
    if not username or not password:
        logger.warning("Bluesky credentials not set. Skipping crawl.")
        return

    try:
        bsky_client.login(username, password)
        crawler = GraphCrawler(bsky_client, social_graph)
        
        # Crawl starting from the bot's own account or a seed list
        # detailed implementation of seed selection can be improved
        crawler.crawl(start_actor=username, depth=2, max_users=500)
        
    except Exception as e:
        logger.error(f"Graph update failed: {e}")
    
    # Save graph stats for UI
    try:
        stats = {
            "node_count": social_graph.number_of_nodes(),
            "edge_count": social_graph.number_of_edges(),
            "last_updated": time.time(),
            "status": "active" if username and password else "configuration_missing"
        }
        
        # Calculate basic community stats for the UI
        try:
            undirected = social_graph.to_undirected()
            if undirected.number_of_nodes() > 0:
                partition = community_louvain.best_partition(undirected)
                community_counts = {}
                for comm_id in partition.values():
                    community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
                
                top_communities = sorted(community_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                stats["top_communities"] = [{"id": cid, "count": count} for cid, count in top_communities]
                stats["total_communities"] = len(community_counts)
        except Exception as e:
            logger.error(f"Failed to calculate community stats: {e}")
            stats["error"] = str(e)

        # Path to state file
        # Assuming we run from project root, or we find it relative to this file
        # This file is in mcp_servers/social_graph/server.py
        # Root is ../../
        root_dir = Path(__file__).parent.parent.parent
        state_file = root_dir / "state" / "social_graph_stats.json"
        
        with open(state_file, "w") as f:
            json.dump(stats, f)
            
        logger.info(f"Graph stats saved to {state_file}")

    except Exception as e:
        logger.error(f"Failed to save graph stats: {e}")

# Start the scheduler
scheduler.add_job(update_graph_job, 'interval', hours=6, id='graph_update')
scheduler.start()

@mcp.tool()
def discover_communities(resolution: float = 1.0) -> str:
    """
    Detects communities in the social graph using the Louvain algorithm.
    Returns a summary of communities found.
    """
    return discover_communities_louvain(social_graph, resolution)

@mcp.tool()
def get_hub_list(community_id: int = None, limit: int = 10) -> str:
    """
    Identifies 'hub' users with high centrality.
    If community_id is provided, filters to that community.
    """
    return get_centrality_hubs(social_graph, community_id, limit)

@mcp.tool()
def force_refresh_graph():
    """
    Triggers an immediate update of the social graph.
    """
    scheduler.get_job('graph_update').func()
    return f"Graph refresh triggered. Current nodes: {social_graph.number_of_nodes()}"

# Run the server
if __name__ == "__main__":
    mcp.run()
