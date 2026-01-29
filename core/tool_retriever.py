"""
DeepAgent Protocol - Story 2.3: Dynamic Tool Retrieval (The Scout)

Provides vector-based/semantic retrieval of tools based on the current step,
optimizing context window usage and reducing token costs.

Migrated from lovev1 and enhanced for Epic 1 with Tool Gap Detection hooks.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from core.semantic_similarity import get_similarity_checker
from core.logger import log_event

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
    
    Story 2.3: Selects tools based on the current step description.
    Epic 1: Signals when no suitable tools are found (Gap Detection).
    """
    
    # Tool categories for keyword-based pre-filtering
    TOOL_CATEGORIES = {
        "image": ["image", "generate", "visual", "picture", "artwork", "photo", "render", "draw"],
        "social": ["post", "tweet", "bluesky", "reply", "notification", "timeline", "social", "like", "repost"],
        "memory": ["memory", "remember", "recall", "store", "save", "retrieve", "forget"],
        "file": ["file", "read", "write", "save", "load", "create", "delete", "directory", "folder"],
        "code": ["code", "execute", "python", "script", "function", "debug", "test", "fabricate"],
        "web": ["fetch", "url", "http", "api", "request", "download", "browse", "search"],
        "analysis": ["analyze", "summarize", "extract", "parse", "evaluate", "assess", "check"]
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
        
        # Gap detection hooks
        self._gap_listeners: List[Callable[[str, float], None]] = []
    
    def add_gap_listener(self, listener: Callable[[str, float], None]):
        """Register a callback for when retrieval yields low confidence."""
        self._gap_listeners.append(listener)
    
    def index_tools(self, registry) -> None:
        """
        Index tools from the registry for efficient retrieval.
        """
        self._tool_cache.clear()
        
        # Get schemas first (which include description)
        schemas = registry.get_schemas()
        
        for schema in schemas:
            name = schema["name"]
            self._tool_cache[name] = {
                "name": name,
                "description": schema.get("description", ""),
                "schema": schema,
                "searchable": self._build_searchable_text(name, schema)
            }
        
        log_event(f"ToolRetriever: Indexed {len(self._tool_cache)} tools", "DEBUG")
    
    def _build_searchable_text(self, name: str, schema: Dict[str, Any]) -> str:
        """Build searchable text from tool name, description and args."""
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
        """
        if not self._tool_cache:
            # If no tools indexed, try keying off global registry
            try:
                from core.tool_registry import get_global_registry
                self.index_tools(get_global_registry())
            except Exception:
                pass
                
        if not self._tool_cache:
            log_event("ToolRetriever: No tools indexed", "WARNING")
            return []
        
        # Step 1: Category-based pre-filtering (fast)
        candidate_tools = self._prefilter_by_category(step_description, include_categories)
        
        if not candidate_tools:
            candidate_tools = list(self._tool_cache.keys())
        
        # Step 2: Semantic similarity scoring
        matches = []
        best_score = 0.0
        
        for tool_name in candidate_tools:
            tool_data = self._tool_cache[tool_name]
            
            score = self.similarity_checker.compute_similarity(
                step_description,
                tool_data["searchable"]
            )
            
            if score > best_score:
                best_score = score
            
            if score >= self.similarity_threshold:
                matches.append(ToolMatch(
                    name=tool_name,
                    description=tool_data["description"],
                    score=score,
                    schema=tool_data["schema"]
                ))
        
        # Sort by score descending
        matches.sort(key=lambda x: x.score, reverse=True)
        result = matches[:max_tools]
        
        # Epic 1: Gap Detection Trigger
        # If no tools matched or best score is low, notify listeners
        if not result or best_score < self.similarity_threshold:
            self._notify_gap(step_description, best_score)
        
        return result
    
    def _notify_gap(self, step_description: str, best_score: float):
        """Notify listeners of potential capability gap."""
        for listener in self._gap_listeners:
            try:
                listener(step_description, best_score)
            except Exception as e:
                logger.error(f"Error in gap listener: {e}")
    
    def _prefilter_by_category(
        self,
        step_description: str,
        include_categories: Optional[List[str]] = None
    ) -> List[str]:
        """Pre-filter tools by category keywords."""
        step_lower = step_description.lower()
        matched_categories = set()
        
        for category, keywords in self.TOOL_CATEGORIES.items():
            if include_categories and category not in include_categories:
                continue
            if any(keyword in step_lower for keyword in keywords):
                matched_categories.add(category)
        
        if not matched_categories:
            return []
        
        candidate_tools = []
        for tool_name, tool_data in self._tool_cache.items():
            # Check if this tool belongs to any matched category
            # We check if the tool's searchable text matches the category keywords
            searchable_lower = tool_data["searchable"].lower()
            for category in matched_categories:
                 matches_category = False
                 for keyword in self.TOOL_CATEGORIES[category]:
                     if keyword in searchable_lower:
                         matches_category = True
                         break
                 if matches_category:
                     candidate_tools.append(tool_name)
                     break
        
        return candidate_tools
    
    def get_tool_menu(self, step_description: str, max_tools: int = 10) -> str:
        """Returns a compact tool list (Story 2.2)."""
        tools = self.retrieve(step_description, max_tools=max_tools)
        
        if not tools:
            return "AVAILABLE TOOLS: None matched. (Tool Gap Detection Active)"
        
        menu_lines = ["AVAILABLE TOOLS (select one to see full API):"]
        for tool in tools:
            desc = tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
            menu_lines.append(f"  [{tool.name}]: {desc}")
            
        return "\n".join(menu_lines)

    def get_full_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Returns full schema for a specific tool."""
        if tool_name not in self._tool_cache:
            return None
        return self._tool_cache[tool_name].get("schema")

# Singleton
_tool_retriever: Optional[ToolRetriever] = None

def get_tool_retriever() -> ToolRetriever:
    """Get the global ToolRetriever instance."""
    global _tool_retriever
    if _tool_retriever is None:
        _tool_retriever = ToolRetriever()
    return _tool_retriever
