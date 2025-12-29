"""
Subagent Loader - Dynamically loads agent prompts from LangChain Hub.

This module provides a robust interface for fetching, caching, and validating
agent prompts from the LangChain Hub, enabling DeepAgent to extend its 
capabilities by dynamically loading specialized personas.
"""
from typing import Optional, Dict, List
from core.prompt_registry import get_prompt_registry
from core.logging import log_event
import os

class SubagentLoader:
    """
    Loads and caches subagent prompts from LangChain Hub.
    
    Features:
    - In-memory caching to prevent redundant API calls
    - Fallback mechanism to local prompts
    - Graceful error handling for offline scenarios
    """
    
    _cache: Dict[str, str] = {}
    
    # Default repository namespace
    DEFAULT_REPO = os.environ.get("LANGCHAIN_HUB_REPO", "love-agent")
    
    @classmethod
    def load_subagent_prompt(cls, agent_type: str, fallback_prompt: str = None) -> str:
        """
        Loads a subagent prompt from the Hub based on agent type.
        
        Args:
            agent_type: The type/role of the agent (e.g., 'coder', 'poet', 'security')
            fallback_prompt: Verification prompt to use if Hub fetch fails.
            
        Returns:
            The loaded prompt string.
        """
        if agent_type in cls._cache:
             log_event(f"Loaded subagent '{agent_type}' from cache", "DEBUG")
             return cls._cache[agent_type]
             
        hub_id = f"{cls.DEFAULT_REPO}/{agent_type}-subagent"
        registry = get_prompt_registry()
        
        log_event(f"Attempting to load subagent '{agent_type}' from Hub: {hub_id}", "INFO")
        
        # Try fetching from Hub via Registry (which handles the Hub client)
        prompt = registry.get_hub_prompt(hub_id)
        
        if prompt:
            cls._cache[agent_type] = prompt
            return prompt
            
        # Fallback logic
        if fallback_prompt:
             log_event(f"Using fallback prompt for subagent '{agent_type}'", "WARNING")
             return fallback_prompt
             
        log_event(f"Failed to load subagent '{agent_type}' and no fallback provided.", "ERROR")
        return f"You are a helpful AI assistant specialized in {agent_type}."

    @classmethod
    def list_cached_subagents(cls) -> List[str]:
        """Lists subagent types currently in cache."""
        return list(cls._cache.keys())
        
    @classmethod
    def clear_cache(cls):
        """Clears the subagent cache."""
        cls._cache.clear()
        
    @classmethod
    def prefetch_common_subagents(cls):
        """Prefetches commonly used subagents to warm the cache."""
        common_types = ["coder", "researcher", "critic"]
        for agent_type in common_types:
            try:
                cls.load_subagent_prompt(agent_type)
            except Exception:
                pass
