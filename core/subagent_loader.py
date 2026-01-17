"""
Subagent Loader - Dynamically loads agent prompts from LangChain Hub.

This module provides a robust interface for fetching, caching, and validating
agent prompts from the LangChain Hub, enabling DeepAgent to extend its 
capabilities by dynamically loading specialized personas.

Features:
- Hub search for capability-based prompt discovery (e.g., "mathematician")
- Separate cache file for discovered prompts (distinct from core prompts.yaml)
- LLM-assisted prompt selection when multiple options available
"""
from typing import Optional, Dict, List, Any
from core.prompt_registry import get_prompt_registry
from core.logging import log_event
from datetime import datetime
import os
import yaml


# Separate cache file for discovered subagent prompts
HUB_PROMPT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hub_prompt_cache.yaml")


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

    # ========================================================================
    # NEW: Hub Search & File-Based Cache for Discovered Prompts
    # ========================================================================
    
    # Expanded capability-to-prompt mapping for dynamic discovery
    CAPABILITY_KEYWORDS = {
        # Math & Science
        "mathematician": ["math", "calculation", "algebra", "calculus"],
        "scientist": ["science", "research", "experiment", "hypothesis"],
        "physicist": ["physics", "quantum", "mechanics", "relativity"],
        "chemist": ["chemistry", "molecule", "reaction", "compound"],
        "biologist": ["biology", "genetics", "cell", "organism"],
        
        # Professional
        "lawyer": ["legal", "law", "contract", "compliance"],
        "accountant": ["accounting", "finance", "tax", "audit"],
        "doctor": ["medical", "health", "diagnosis", "treatment"],
        "engineer": ["engineering", "design", "system", "build"],
        
        # Creative
        "writer": ["writing", "story", "narrative", "prose"],
        "poet": ["poetry", "verse", "rhyme", "lyric"],
        "musician": ["music", "composition", "melody", "harmony"],
        "artist": ["art", "visual", "design", "aesthetic"],
        
        # Technical
        "programmer": ["code", "programming", "software", "development"],
        "data_scientist": ["data", "analytics", "statistics", "ml"],
        "security_expert": ["security", "vulnerability", "penetration", "audit"],
        "devops": ["infrastructure", "deployment", "ci/cd", "cloud"],
    }
    
    # Additional Hub prompts mapped to capabilities
    EXTENDED_HUB_PROMPTS = {
        "mathematician": "hwchase17/structured-chat-agent",
        "scientist": "hwchase17/react",
        "lawyer": "hwchase17/structured-chat-agent",
        "writer": "hwchase17/react",
        "programmer": "hwchase17/structured-chat-agent",
        "data_scientist": "hwchase17/structured-chat-agent",
    }
    
    @classmethod
    async def search_hub_for_capability(
        cls, 
        capability: str, 
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search LangChain Hub for prompts matching a desired capability.
        
        This enables truly dynamic subagent creation - agents can describe
        what they need (e.g., "mathematician", "legal expert") and the system
        finds appropriate prompts.
        
        Args:
            capability: Description of the capability needed (e.g., "mathematician")
            max_results: Maximum results to return
            
        Returns:
            List of {id, name, description, score} for matching prompts
        """
        capability_lower = capability.lower()
        results = []
        
        log_event(f"Searching Hub for capability: {capability}", "INFO")
        
        # Step 1: Check file-based cache first
        cached = cls._load_from_file_cache(capability_lower)
        if cached:
            log_event(f"Found cached prompt for '{capability}'", "DEBUG")
            return [{
                "id": cached.get("hub_id", f"cached/{capability_lower}"),
                "name": capability.title(),
                "description": f"Cached prompt for {capability}",
                "score": 1.0,
                "content": cached.get("content"),
                "source": "file_cache"
            }]
        
        # Step 2: Match against known capability keywords
        for cap_name, keywords in cls.CAPABILITY_KEYWORDS.items():
            if cap_name in capability_lower or any(kw in capability_lower for kw in keywords):
                # Found a match - get the associated Hub prompt
                hub_id = cls.EXTENDED_HUB_PROMPTS.get(cap_name, "hwchase17/react")
                
                results.append({
                    "id": hub_id,
                    "name": cap_name.replace("_", " ").title(),
                    "description": f"Agent specialized in {cap_name.replace('_', ' ')}",
                    "score": 0.9 if cap_name in capability_lower else 0.7,
                    "source": "keyword_match"
                })
        
        # Step 3: Try direct Hub lookup if we have langchainhub
        try:
            from langchain import hub
            
            # Try well-known public prompts
            public_options = [
                ("hwchase17/react", "ReAct reasoning agent"),
                ("hwchase17/structured-chat-agent", "Structured chat with tools"),
                ("hwchase17/openai-functions-agent", "OpenAI function calling"),
            ]
            
            for hub_id, desc in public_options:
                if hub_id not in [r["id"] for r in results]:
                    results.append({
                        "id": hub_id,
                        "name": hub_id.split("/")[-1].replace("-", " ").title(),
                        "description": desc,
                        "score": 0.5,  # Lower score for generic prompts
                        "source": "hub_public"
                    })
                    
        except ImportError:
            log_event("langchainhub not available for direct Hub search", "DEBUG")
        
        # Step 4: Add LOVE's own prompts as fallback
        love_prompt_id = f"{cls.DEFAULT_REPO}/{capability_lower.replace(' ', '-')}-subagent"
        results.append({
            "id": love_prompt_id,
            "name": f"LOVE {capability.title()}",
            "description": f"L.O.V.E. specialized {capability} agent",
            "score": 0.6,
            "source": "love_repo"
        })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        log_event(f"Found {len(results)} prompts for '{capability}'", "INFO")
        return results[:max_results]
    
    @classmethod
    async def get_best_prompt_for_capability(cls, capability: str) -> Optional[str]:
        """
        Get the best prompt for a given capability, fetching from Hub if needed.
        
        This is the primary entry point for dynamic subagent creation:
        1. Search for matching prompts
        2. Fetch the best one from Hub
        3. Cache it for future use
        
        Args:
            capability: Description of what the agent should do
            
        Returns:
            The prompt string, or None if not found
        """
        # Search for matching prompts
        search_results = await cls.search_hub_for_capability(capability)
        
        if not search_results:
            log_event(f"No Hub prompts found for '{capability}'", "WARNING")
            return None
        
        # Try each result until one succeeds
        registry = get_prompt_registry()
        
        for result in search_results:
            # If already has content (from file cache), use it
            if result.get("content"):
                return result["content"]
            
            hub_id = result["id"]
            log_event(f"Fetching prompt from Hub: {hub_id}", "INFO")
            
            try:
                prompt = registry.get_hub_prompt(hub_id)
                
                if prompt:
                    # Cache to file for future use
                    cls._save_to_file_cache(capability, hub_id, prompt)
                    return prompt
                    
            except Exception as e:
                log_event(f"Failed to fetch {hub_id}: {e}", "DEBUG")
                continue
        
        log_event(f"All Hub fetches failed for '{capability}'", "WARNING")
        return None
    
    @classmethod
    def _load_from_file_cache(cls, capability: str) -> Optional[Dict[str, Any]]:
        """Load a prompt from the file-based cache."""
        if not os.path.exists(HUB_PROMPT_CACHE_FILE):
            return None
        
        try:
            with open(HUB_PROMPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = yaml.safe_load(f) or {}
            
            # Normalize the capability key
            cap_key = capability.lower().replace(" ", "_")
            
            if cap_key in cache_data:
                return cache_data[cap_key]
                
        except Exception as e:
            log_event(f"Error reading Hub prompt cache: {e}", "DEBUG")
        
        return None
    
    @classmethod
    def _save_to_file_cache(cls, capability: str, hub_id: str, content: str):
        """Save a prompt to the file-based cache (separate from core prompts)."""
        try:
            # Load existing cache
            cache_data = {}
            if os.path.exists(HUB_PROMPT_CACHE_FILE):
                with open(HUB_PROMPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = yaml.safe_load(f) or {}
            
            # Normalize the capability key
            cap_key = capability.lower().replace(" ", "_")
            
            # Add new entry
            cache_data[cap_key] = {
                "content": content,
                "hub_id": hub_id,
                "capability": capability,
                "fetched_at": datetime.now().isoformat(),
            }
            
            # Write back (this is SEPARATE from core prompts.yaml)
            with open(HUB_PROMPT_CACHE_FILE, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cache_data, f, indent=2, width=4096, allow_unicode=True, default_flow_style=False)
            
            log_event(f"Cached Hub prompt for '{capability}' to {HUB_PROMPT_CACHE_FILE}", "INFO")
            
        except Exception as e:
            log_event(f"Failed to cache Hub prompt: {e}", "WARNING")
    
    @classmethod
    def list_file_cached_prompts(cls) -> List[str]:
        """List all prompts in the file-based cache."""
        if not os.path.exists(HUB_PROMPT_CACHE_FILE):
            return []
        
        try:
            with open(HUB_PROMPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = yaml.safe_load(f) or {}
            return list(cache_data.keys())
        except Exception:
            return []

