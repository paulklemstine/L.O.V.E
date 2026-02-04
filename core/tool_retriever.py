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
    
    Enhanced for Open Agentic Web:
    - ObjectIndex pattern: Stores actual callable tool objects for retrieval
    - Cross-registry retrieval: Combines local tools + MCP tools
    - Filesystem discovery: Lazy exploration of tool directories
    """
    
    # Tool categories for keyword-based pre-filtering
    TOOL_CATEGORIES = {
        "image": ["image", "generate", "visual", "picture", "artwork", "photo", "render", "draw"],
        "social": ["post", "tweet", "bluesky", "reply", "notification", "timeline", "social", "like", "repost"],
        "memory": ["memory", "remember", "recall", "store", "save", "retrieve", "forget"],
        "file": ["file", "read", "write", "save", "load", "create", "delete", "directory", "folder"],
        "code": ["code", "execute", "python", "script", "function", "debug", "test", "fabricate"],
        "web": ["fetch", "url", "http", "api", "request", "download", "browse", "search"],
        "analysis": ["analyze", "summarize", "extract", "parse", "evaluate", "assess", "check"],
        "mcp": ["mcp", "server", "protocol", "context", "registry"]
    }
    
    def __init__(self, similarity_threshold: float = 0.3, include_mcp: bool = True):
        """
        Initialize the tool retriever.
        
        Args:
            similarity_threshold: Minimum similarity score for tool selection (0.0-1.0)
            include_mcp: Whether to include MCP tools in retrieval
        """
        self.similarity_threshold = similarity_threshold
        self.include_mcp = include_mcp
        self.similarity_checker = get_similarity_checker()
        
        # Standard tool cache (metadata only)
        self._tool_cache: Dict[str, Dict[str, Any]] = {}
        
        # ObjectIndex: Stores actual callable tool objects (LlamaIndex pattern)
        self._object_index: Dict[str, Any] = {}  # name -> actual tool object
        
        # MCP tool cache (from dynamic discovery)
        self._mcp_tool_cache: Dict[str, Dict[str, Any]] = {}
        
        # Filesystem tool directories for lazy discovery
        self._tool_directories: List[str] = []
        
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
            
            # Story 2.3: Fallback for Small Toolsets (e.g., Colab, fresh install)
            # If we have a small number of tools and retrieval failed, 
            # it's better to show everything than nothing.
            total_tools = len(self._tool_cache)
            if not result and total_tools <= 20:
                log_event(f"ToolRetriever: Fallback to all {total_tools} tools due to low similarity.", "INFO")
                fallback_matches = []
                for tool_name, tool_data in self._tool_cache.items():
                    fallback_matches.append(ToolMatch(
                        name=tool_name,
                        description=tool_data["description"],
                        score=0.1,  # Low score to indicate fallback
                        schema=tool_data["schema"]
                    ))
                # Sort by name for stability
                fallback_matches.sort(key=lambda x: x.name)
                return fallback_matches
        
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
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name].get("schema")
        if tool_name in self._mcp_tool_cache:
            return self._mcp_tool_cache[tool_name].get("schema")
        return None
    
    # =========================================================================
    # ObjectIndex Pattern (LlamaIndex-style)
    # =========================================================================
    
    def index_as_objects(self, tools: Dict[str, Any]) -> None:
        """
        Index actual tool objects for retrieval (ObjectIndex pattern).
        
        Unlike index_tools which stores metadata, this stores the actual
        callable tool objects, enabling direct retrieval and execution.
        
        Args:
            tools: Dict mapping tool names to callable tool objects
        """
        for name, tool_obj in tools.items():
            # Store the actual object
            self._object_index[name] = tool_obj
            
            # Extract metadata for similarity search
            description = ""
            schema = {}
            
            if hasattr(tool_obj, 'description'):
                description = tool_obj.description
            if hasattr(tool_obj, 'schema'):
                schema = tool_obj.schema if callable(tool_obj.schema) else tool_obj.schema
            if hasattr(tool_obj, '__doc__') and not description:
                description = tool_obj.__doc__ or ""
            
            # Add to tool cache for similarity search
            self._tool_cache[name] = {
                "name": name,
                "description": description,
                "schema": schema,
                "searchable": self._build_searchable_text(name, {"description": description}),
                "is_object": True
            }
        
        log_event(f"ToolRetriever: Indexed {len(tools)} tool objects", "DEBUG")
    
    def retrieve_objects(
        self,
        step_description: str,
        max_tools: int = 5
    ) -> List[Any]:
        """
        Retrieve actual tool objects (not just metadata).
        
        This is the ObjectRetriever pattern from LlamaIndex.
        
        Args:
            step_description: Description of what the tool should do
            max_tools: Maximum number of tools to return
            
        Returns:
            List of actual callable tool objects
        """
        matches = self.retrieve(step_description, max_tools=max_tools)
        objects = []
        
        for match in matches:
            if match.name in self._object_index:
                objects.append(self._object_index[match.name])
        
        return objects
    
    # =========================================================================
    # MCP Cross-Registry Retrieval
    # =========================================================================
    
    def index_mcp_tools(self, server_name: str, tools: Dict[str, str]) -> None:
        """
        Index tools from an MCP server for retrieval.
        
        Args:
            server_name: Name of the MCP server
            tools: Dict mapping tool names to descriptions
        """
        for tool_name, description in tools.items():
            full_name = f"{server_name}.{tool_name}"
            self._mcp_tool_cache[full_name] = {
                "name": full_name,
                "server": server_name,
                "tool": tool_name,
                "description": description,
                "schema": {"name": tool_name, "description": description},
                "searchable": self._build_searchable_text(full_name, {"description": description})
            }
        
        log_event(f"ToolRetriever: Indexed {len(tools)} MCP tools from {server_name}", "DEBUG")
    
    def retrieve_with_mcp(
        self,
        step_description: str,
        max_tools: int = 5
    ) -> Tuple[List[ToolMatch], List[ToolMatch]]:
        """
        Retrieve from both local registry and MCP servers.
        
        Args:
            step_description: Description of what the tool should do
            max_tools: Maximum number of tools from each source
            
        Returns:
            Tuple of (local_matches, mcp_matches)
        """
        # Get local matches
        local_matches = self.retrieve(step_description, max_tools=max_tools)
        
        # Get MCP matches
        mcp_matches = []
        best_score = 0.0
        
        for tool_name, tool_data in self._mcp_tool_cache.items():
            score = self.similarity_checker.compute_similarity(
                step_description,
                tool_data["searchable"]
            )
            
            if score > best_score:
                best_score = score
            
            if score >= self.similarity_threshold:
                mcp_matches.append(ToolMatch(
                    name=tool_name,
                    description=tool_data["description"],
                    score=score,
                    schema=tool_data["schema"]
                ))
        
        mcp_matches.sort(key=lambda x: x.score, reverse=True)
        
        return local_matches, mcp_matches[:max_tools]
    
    # =========================================================================
    # Filesystem Discovery ("ls" Pattern)
    # =========================================================================
    
    def add_tool_directory(self, path: str) -> None:
        """
        Add a directory to explore for tool discovery.
        
        Implements the "filesystem as context" pattern where tools are
        stored as Python files and discovered lazily.
        
        Args:
            path: Absolute path to a directory containing tool files
        """
        import os
        if os.path.isdir(path) and path not in self._tool_directories:
            self._tool_directories.append(path)
            log_event(f"ToolRetriever: Added tool directory {path}", "DEBUG")
    
    def filesystem_discovery(
        self,
        query: str,
        max_depth: int = 3
    ) -> List[Dict[str, str]]:
        """
        Explore tool directories to find relevant tools.
        
        The 'ls' pattern - explores the tool hierarchy like a decision tree,
        reducing token consumption by ~98.7% compared to loading all tools.
        
        Args:
            query: What kind of tool to look for
            max_depth: Maximum directory depth to explore
            
        Returns:
            List of dicts with 'path' and 'description' for matching files
        """
        import os
        
        matches = []
        query_lower = query.lower()
        
        for base_dir in self._tool_directories:
            for root, dirs, files in os.walk(base_dir):
                # Check depth
                depth = root[len(base_dir):].count(os.sep)
                if depth > max_depth:
                    dirs.clear()  # Don't descend further
                    continue
                
                for file in files:
                    if not file.endswith('.py') or file.startswith('_'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    file_name = file[:-3]  # Remove .py
                    
                    # Quick check: does filename match query?
                    if query_lower in file_name.lower():
                        matches.append({
                            "path": file_path,
                            "name": file_name,
                            "description": f"Tool module: {file_name}"
                        })
                        continue
                    
                    # Check file contents (first few lines)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            header = f.read(500)
                            if query_lower in header.lower():
                                # Extract docstring if present
                                desc = self._extract_docstring(header)
                                matches.append({
                                    "path": file_path,
                                    "name": file_name,
                                    "description": desc or f"Tool module: {file_name}"
                                })
                    except:
                        pass
        
        return matches[:10]  # Limit results
    
    def _extract_docstring(self, content: str) -> Optional[str]:
        """Extract module docstring from file content."""
        import re
        
        # Match triple-quoted docstring at start
        match = re.search(r'^["\'][\'"]{2}(.*?)["\'][\'"]{2}', content, re.DOTALL)
        if match:
            return match.group(1).strip()[:100]
        return None

# Singleton
_tool_retriever: Optional[ToolRetriever] = None

def get_tool_retriever() -> ToolRetriever:
    """Get the global ToolRetriever instance."""
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
    return get_tool_retriever().get_tool_menu(step_description)  # Use tool menu as subset metadata

