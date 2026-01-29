"""
Tools for Influencer Scouting and Interaction.
Auto-discovered by ToolRegistry.
"""
from core.agents.influencer_scout_agent import influencer_scout_agent
from core.tool_registry import tool_schema
from typing import Dict, Any, List

@tool_schema
def scout_influencers(start_user: str = "bsky.app", depth: int = 1) -> Dict[str, Any]:
    """
    Scout the Bluesky social graph to find influential users.
    
    Args:
        start_user: User handle to start traversing from (seed).
        depth: How many degrees of separation to traverse (1-3).
    
    Returns:
        Dict with stats on discovered users.
    """
    import asyncio
    # Helper to run async method synchronously
    from core.bluesky_agent import _run_sync_safe
    
    return _run_sync_safe(
        influencer_scout_agent.scout_network(seed_users=[start_user], depth=depth)
    )

@tool_schema
def engage_top_influencer(dry_run: bool = True) -> Dict[str, Any]:
    """
    Selects the highest scoring influencer and posts a reply to them.
    Respects "one per day" limit per user.
    
    Args:
        dry_run: If True, simulates the engagement without posting.
        
    Returns:
        Dict with engagement results (target, text, success).
    """
    from core.bluesky_agent import _run_sync_safe
    
    return _run_sync_safe(
        influencer_scout_agent.engage_influencer(dry_run=dry_run)
    )
