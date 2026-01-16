"""
Subagent Loader - Dynamically loads agent prompts from LangChain Hub.

This module provides a robust interface for fetching, caching, and validating
agent prompts from the LangChain Hub, enabling DeepAgent to extend its 
capabilities by dynamically loading specialized personas.
"""
from typing import Optional, Dict, List, Any
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
    - LangHub discovery for available agent prompts
    """
    
    _cache: Dict[str, str] = {}
    _hub_cache: Dict[str, str] = {}  # Separate cache for direct Hub loads
    
    # Default repository namespace
    DEFAULT_REPO = os.environ.get("LANGCHAIN_HUB_REPO", "love-agent")
    
    # Known high-quality public Hub prompts for various agent types
    PUBLIC_HUB_PROMPTS = {
        "reasoning": "hwchase17/react",
        "coding": "hwchase17/structured-chat-agent",
        "research": "hwchase17/react",
        "analyst": "hwchase17/structured-chat-agent",
    }
    
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
        
        # Try public Hub prompts
        if agent_type in cls.PUBLIC_HUB_PROMPTS:
            public_id = cls.PUBLIC_HUB_PROMPTS[agent_type]
            log_event(f"Trying public Hub prompt: {public_id}", "DEBUG")
            public_prompt = registry.get_hub_prompt(public_id)
            if public_prompt:
                cls._cache[agent_type] = public_prompt
                return public_prompt
            
        # Fallback logic
        if fallback_prompt:
             log_event(f"Using fallback prompt for subagent '{agent_type}'", "WARNING")
             return fallback_prompt
             
        log_event(f"Failed to load subagent '{agent_type}' and no fallback provided.", "ERROR")
        return f"You are a helpful AI assistant specialized in {agent_type}."

    @classmethod
    def load_from_hub(cls, hub_id: str) -> Optional[str]:
        """
        Loads a specific prompt from LangChain Hub by full ID.
        
        Args:
            hub_id: The full Hub ID (e.g., "hwchase17/react" or "love-agent/coder")
            
        Returns:
            The prompt string or None if not found
        """
        if hub_id in cls._hub_cache:
            log_event(f"Loaded Hub prompt '{hub_id}' from cache", "DEBUG")
            return cls._hub_cache[hub_id]
        
        registry = get_prompt_registry()
        
        log_event(f"Loading prompt from Hub: {hub_id}", "INFO")
        
        try:
            prompt = registry.get_hub_prompt(hub_id)
            if prompt:
                cls._hub_cache[hub_id] = prompt
                return prompt
        except Exception as e:
            log_event(f"Failed to load Hub prompt {hub_id}: {e}", "WARNING")
        
        return None

    @classmethod
    async def discover_hub_agents(cls) -> List[Dict[str, Any]]:
        """
        Discovers available agent prompts from LangChain Hub.
        
        This returns LOVE's own published agents plus some well-known
        public agent prompts.
        
        Returns:
            List of {id, name, description, source} for available agents
        """
        agents = []
        
        # Add known public high-quality prompts
        public_agents = [
            {
                "id": "hwchase17/react",
                "name": "ReAct Agent",
                "description": "Reasoning and Acting agent using chain-of-thought",
                "source": "langhub_public"
            },
            {
                "id": "hwchase17/structured-chat-agent",
                "name": "Structured Chat Agent",
                "description": "Structured conversational agent for complex tasks",
                "source": "langhub_public"
            },
            {
                "id": "hwchase17/openai-functions-agent",
                "name": "OpenAI Functions Agent",
                "description": "Agent using OpenAI function calling",
                "source": "langhub_public"
            },
        ]
        agents.extend(public_agents)
        
        # Try to list LOVE's own published agents
        love_agents = [
            {
                "id": f"{cls.DEFAULT_REPO}/coder-subagent",
                "name": "LOVE Coder",
                "description": "Code generation and modification specialist",
                "source": "langhub_love"
            },
            {
                "id": f"{cls.DEFAULT_REPO}/researcher-subagent",
                "name": "LOVE Researcher",
                "description": "Information gathering and synthesis",
                "source": "langhub_love"
            },
            {
                "id": f"{cls.DEFAULT_REPO}/critic-subagent",
                "name": "LOVE Critic",
                "description": "Code and plan review specialist",
                "source": "langhub_love"
            },
            {
                "id": f"{cls.DEFAULT_REPO}/creative-subagent",
                "name": "LOVE Creative",
                "description": "Creative content generation",
                "source": "langhub_love"
            },
        ]
        agents.extend(love_agents)
        
        return agents

    @classmethod
    def list_cached_subagents(cls) -> List[str]:
        """Lists subagent types currently in cache."""
        return list(cls._cache.keys())
    
    @classmethod
    def list_cached_hub_prompts(cls) -> List[str]:
        """Lists Hub prompt IDs currently in cache."""
        return list(cls._hub_cache.keys())
        
    @classmethod
    def clear_cache(cls):
        """Clears all caches."""
        cls._cache.clear()
        cls._hub_cache.clear()
        log_event("SubagentLoader cache cleared", "DEBUG")
        
    @classmethod
    def prefetch_common_subagents(cls):
        """Prefetches commonly used subagents to warm the cache."""
        common_types = ["coder", "researcher", "critic", "reasoning", "creative"]
        for agent_type in common_types:
            try:
                cls.load_subagent_prompt(agent_type)
            except Exception as e:
                log_event(f"Failed to prefetch {agent_type}: {e}", "DEBUG")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get statistics about cached prompts."""
        return {
            "subagent_cache_size": len(cls._cache),
            "hub_cache_size": len(cls._hub_cache),
            "total_cached": len(cls._cache) + len(cls._hub_cache),
        }

