import logging
import time
import networkx as nx
from atproto import Client
from atproto.exceptions import AtProtocolError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("social_graph_crawler")

class GraphCrawler:
    def __init__(self, client: Client, graph: nx.Graph):
        self.client = client
        self.graph = graph

    def crawl(self, start_actor: str, depth: int = 1, max_users: int = 100):
        """
        Crawls the social graph starting from a seed user.
        
        Args:
            start_actor (str): The handle or DID of the user to start crawling from.
            depth (int): The depth of the crawl (1 = just the user's follows, 2 = user's follows' follows, etc.)
            max_users (int): The maximum number of users to visit to prevent infinite crawling.
        """
        logger.info(f"Starting crawl from {start_actor} with depth {depth}")
        
        visited = set()
        queue = [(start_actor, 0)] # (actor_did, current_depth)
        
        count = 0
        
        try:
            # Resolve start actor to DID if it's a handle
            if not start_actor.startswith("did:"):
                profile = self.client.get_profile(actor=start_actor)
                start_did = profile.did
                # Update queue with DID
                queue[0] = (start_did, 0)
                # Add node for start user
                self.graph.add_node(start_did, handle=profile.handle, display_name=profile.display_name)
            else:
                start_did = start_actor
                # Try to get profile to enrich node data
                try:
                    profile = self.client.get_profile(actor=start_did)
                    self.graph.add_node(start_did, handle=profile.handle, display_name=profile.display_name)
                except Exception:
                    self.graph.add_node(start_did)

        except Exception as e:
            logger.error(f"Failed to resolve start actor {start_actor}: {e}")
            return

        while queue and count < max_users:
            current_did, current_depth = queue.pop(0)
            
            if current_did in visited:
                continue
            
            visited.add(current_did)
            count += 1
            
            if current_depth >= depth:
                continue
                
            logger.info(f"Crawling {current_did} ({count}/{max_users})")
            
            try:
                # Fetch follows
                cursor = None
                while True:
                    # Limit to 100 follows per user for now to be polite and fast
                    response = self.client.get_follows(actor=current_did, cursor=cursor, limit=100)
                    
                    for follow in response.follows:
                        target_did = follow.did
                        
                        # Add node for target
                        if not self.graph.has_node(target_did):
                            self.graph.add_node(target_did, handle=follow.handle, display_name=follow.display_name)
                        
                        # Add edge
                        self.graph.add_edge(current_did, target_did)
                        
                        # Add to queue for next level
                        if target_did not in visited:
                            queue.append((target_did, current_depth + 1))
                    
                    if not response.cursor:
                        break
                    
                    # Stop after one page for now to speed up exploration
                    break 
                    
            except AtProtocolError as e:
                logger.error(f"Error fetching follows for {current_did}: {e}")
                # Simple backoff or continue
                continue
            except Exception as e:
                logger.error(f"Unexpected error crawling {current_did}: {e}")
                continue
                
        logger.info(f"Crawl complete. Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
