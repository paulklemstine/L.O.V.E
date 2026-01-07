"""
DeepAgent Protocol - Story 2.3: Dynamic Tool Retrieval (The Scout)

Provides vector-based/semantic retrieval of tools based on the current step,
optimizing context window usage and reducing token costs.

Uses TF-IDF semantic similarity from core/semantic_similarity.py for
lightweight, dependency-free operation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.semantic_similarity import SemanticSimilarityChecker, get_similarity_checker
from core.logging import log_event

logger = logging.getLogger(__name__)


@dataclass
class ToolMatch:
    """Represents a matched tool with its relevance score."""
    name: str
    description: str
    score: float
    schema: Dict[str, Any]


class ToolRetriever:
    """
    Dynamic tool retrieval using semantic similarity.
    
    Story 2.3: Selects tools based on the current step description
    rather than loading all tools at once, optimizing context usage.
    """
    
    # Tool categories for keyword-based pre-filtering
    TOOL_CATEGORIES = {
        "image": ["image", "generate", "visual", "picture", "artwork", "photo", "render"],
        "social": ["post", "tweet", "bluesky", "reply", "notification", "timeline", "social"],
        "memory": ["memory", "remember", "recall", "store", "save", "retrieve", "forget"],
        "file": ["file", "read", "write", "save", "load", "create", "delete", "directory"],
        "code": ["code", "execute", "python", "script", "function", "debug", "test"],
        "web": ["fetch", "url", "http", "api", "request", "download", "browse"],
        "analysis": ["analyze", "summarize", "extract", "parse", "evaluate", "assess"]
    }
    
    def __init__(self, similarity_threshold: float = 0.3):
        """
        Initialize the tool retriever.
        
        Args:
            similarity_threshold: Minimum similarity score for tool selection (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_checker = get_similarity_checker()
        self._tool_cache: Dict[str, Dict[str, Any]] = {}
    
    def index_tools(self, registry) -> None:
        """
        Index tools from the registry for efficient retrieval.
        
        Args:
            registry: ToolRegistry instance containing registered tools
        """
        self._tool_cache.clear()
        
        for name in registry.list_tools():
            schema = registry.get_schema(name)
            if schema:
                self._tool_cache[name] = {
                    "name": name,
                    "description": schema.get("description", ""),
                    "schema": schema,
                    # Pre-compute searchable text
                    "searchable": self._build_searchable_text(name, schema)
                }
        
        log_event(f"ToolRetriever: Indexed {len(self._tool_cache)} tools", "DEBUG")
    
    def _build_searchable_text(self, name: str, schema: Dict[str, Any]) -> str:
        """
        Build searchable text from tool name and schema.
        
        Combines name, description, and parameter names for better matching.
        """
        parts = [
            name.replace("_", " "),
            schema.get("description", "")
        ]
        
        # Add parameter names and descriptions
        params = schema.get("parameters", {}).get("properties", {})
        for param_name, param_info in params.items():
            parts.append(param_name.replace("_", " "))
            if isinstance(param_info, dict):
                parts.append(param_info.get("description", ""))
        
        return " ".join(parts)
    
    def retrieve(
        self,
        step_description: str,
        max_tools: int = 5,
        include_categories: Optional[List[str]] = None
    ) -> List[ToolMatch]:
        """
        Retrieve relevant tools for a given step.
        
        Story 2.3: Uses semantic similarity to select only the tools
        relevant to the current step, reducing context window usage.
        
        Args:
            step_description: The current step or task description
            max_tools: Maximum number of tools to return
            include_categories: Optional list of category filters
            
        Returns:
            List of ToolMatch objects sorted by relevance
        """
        if not self._tool_cache:
            log_event("ToolRetriever: No tools indexed", "WARNING")
            return []
        
        # Step 1: Category-based pre-filtering (fast)
        candidate_tools = self._prefilter_by_category(step_description, include_categories)
        
        if not candidate_tools:
            # Fall back to all tools if no category match
            candidate_tools = list(self._tool_cache.keys())
        
        # Step 2: Semantic similarity scoring
        matches = []
        for tool_name in candidate_tools:
            tool_data = self._tool_cache[tool_name]
            
            # Compute similarity between step and tool's searchable text
            score = self.similarity_checker.compute_similarity(
                step_description,
                tool_data["searchable"]
            )
            
            if score >= self.similarity_threshold:
                matches.append(ToolMatch(
                    name=tool_name,
                    description=tool_data["description"],
                    score=score,
                    schema=tool_data["schema"]
                ))
        
        # Sort by score descending
        matches.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        result = matches[:max_tools]
        
        if result:
            tool_names = [m.name for m in result]
            log_event(
                f"ToolRetriever: Selected {len(result)} tools for step: {tool_names}",
                "DEBUG"
            )
        
        return result
    
    def _prefilter_by_category(
        self,
        step_description: str,
        include_categories: Optional[List[str]] = None
    ) -> List[str]:
        """
        Pre-filter tools by category keywords.
        
        Fast keyword matching to reduce the semantic similarity search space.
        """
        step_lower = step_description.lower()
        matched_categories = set()
        
        # Find matching categories
        for category, keywords in self.TOOL_CATEGORIES.items():
            if include_categories and category not in include_categories:
                continue
            if any(keyword in step_lower for keyword in keywords):
                matched_categories.add(category)
        
        if not matched_categories:
            return []
        
        # Find tools matching these categories
        candidate_tools = []
        for tool_name, tool_data in self._tool_cache.items():
            searchable_lower = tool_data["searchable"].lower()
            for category in matched_categories:
                if any(keyword in searchable_lower for keyword in self.TOOL_CATEGORIES[category]):
                    candidate_tools.append(tool_name)
                    break
        
        return candidate_tools
    
    def get_tool_subset_metadata(self, tools: List[ToolMatch]) -> str:
        """
        Format a subset of tools for LLM prompt injection.
        
        Much smaller than the full tool list, optimizing context usage.
        
        Args:
            tools: List of ToolMatch objects from retrieve()
            
        Returns:
            Formatted string for prompt injection
        """
        if not tools:
            return "No specific tools available for this step."
        
        output = f"Available tools for this step ({len(tools)} selected):\n\n"
        
        for match in tools:
            output += f"**{match.name}** (relevance: {match.score:.2f})\n"
            output += f"  Description: {match.description}\n"
            
            params = match.schema.get("parameters", {}).get("properties", {})
            if params:
                param_names = list(params.keys())[:5]  # Limit shown params
                output += f"  Parameters: {', '.join(param_names)}\n"
            
            output += "\n"
        
        return output


# =============================================================================
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# =============================================================================

_tool_retriever: Optional[ToolRetriever] = None


def get_tool_retriever() -> ToolRetriever:
    """Get or create the global ToolRetriever instance."""
    global _tool_retriever
    if _tool_retriever is None:
        _tool_retriever = ToolRetriever()
    return _tool_retriever


def retrieve_tools_for_step(
    step_description: str,
    registry=None,
    max_tools: int = 5
) -> List[ToolMatch]:
    """
    Convenience function to retrieve tools for a step.
    
    Args:
        step_description: The current step description
        registry: Optional ToolRegistry (will use global if not provided)
        max_tools: Maximum tools to return
        
    Returns:
        List of ToolMatch objects
    """
    retriever = get_tool_retriever()
    
    # Re-index if registry provided and cache is empty
    if registry and not retriever._tool_cache:
        retriever.index_tools(registry)
    elif not retriever._tool_cache:
        # Try to get global registry
        try:
            from core.tool_registry import get_global_registry
            global_registry = get_global_registry()
            retriever.index_tools(global_registry)
        except Exception as e:
            logger.warning(f"Could not index tools: {e}")
            return []
    
    return retriever.retrieve(step_description, max_tools=max_tools)


def format_tools_for_step(step_description: str, registry=None) -> str:
    """
    Get formatted tool metadata for a specific step.
    
    Story 2.3: Returns only relevant tools instead of the full list.
    
    Args:
        step_description: The current step description
        registry: Optional ToolRegistry
        
    Returns:
        Formatted string of relevant tools
    """
    tools = retrieve_tools_for_step(step_description, registry)
    return get_tool_retriever().get_tool_subset_metadata(tools)
