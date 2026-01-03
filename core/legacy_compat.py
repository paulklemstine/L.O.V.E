"""
Legacy Compatibility Layer for Tool Registry.

This module provides backward compatibility for code using the old
ToolRegistry interface from tools_legacy.py while migrating to the
new tool_registry.py implementation.

Usage:
    # Old code:
    from core.tools_legacy import ToolRegistry, SecureExecutor
    
    # New code (with compatibility):
    from core.legacy_compat import LegacyToolRegistry as ToolRegistry
    from core.secure_executor import SecureExecutor
"""
from typing import Any, Callable, Dict
from core.tool_registry import ToolRegistry as NewToolRegistry, tool_schema


class LegacyToolRegistry:
    """
    A compatibility wrapper that provides the old ToolRegistry interface
    while using the new implementation internally.
    
    This allows gradual migration of code from tools_legacy.py to
    the new tool_registry.py system.
    
    Old interface methods:
    - register_tool(name, tool, metadata)
    - get_tool(name)
    - list_tools()
    - get_tool_names()
    - get_formatted_tool_metadata()
    
    New interface also available:
    - register(func, name)
    - get_schemas()
    - get_all_tool_schemas()
    """
    
    def __init__(self):
        self._new_registry = NewToolRegistry()
        self._legacy_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, tool: Callable, metadata: Dict[str, Any]) -> None:
        """
        Legacy registration method that accepts explicit metadata.
        
        Args:
            name: The name of the tool
            tool: The callable to register
            metadata: Dictionary with 'description' and 'arguments' keys
        """
        # Store legacy metadata
        self._legacy_metadata[name] = metadata
        
        # Create a schema-compatible wrapper
        description = metadata.get("description", "No description")
        arguments = metadata.get("arguments", {})
        
        # Build schema in new format
        schema = {
            "name": name,
            "description": description,
            "parameters": arguments
        }
        
        # Attach schema to function
        tool.__tool_schema__ = schema
        
        # Register with new registry
        try:
            self._new_registry.register(tool, name=name)
        except Exception as e:
            # Fallback: register with docstring
            if not tool.__doc__:
                tool.__doc__ = description
            self._new_registry._tools[name] = {
                "func": tool,
                "schema": schema
            }
            print(f"Tool '{name}' registered (legacy mode).")
    
    def register(self, func: Callable, name: str = None) -> None:
        """New-style registration (forwards to new registry)."""
        self._new_registry.register(func, name=name)
    
    def get_tool(self, name: str) -> Callable:
        """Retrieves a tool's callable function by name."""
        return self._new_registry.get_tool(name)
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary of all registered tools and their metadata.
        (Legacy format for compatibility)
        """
        result = {}
        for name, data in self._new_registry._tools.items():
            result[name] = {
                "tool": data["func"],
                "metadata": self._legacy_metadata.get(name, {
                    "description": data["schema"].get("description", ""),
                    "arguments": data["schema"].get("parameters", {})
                })
            }
        return result
    
    def get_tool_names(self) -> list:
        """Returns a list of all registered tool names."""
        return self._new_registry.list_tools()
    
    def get_formatted_tool_metadata(self) -> str:
        """Returns formatted metadata for prompt injection."""
        return self._new_registry.get_formatted_tool_metadata()
    
    def get_schemas(self):
        """New-style: Returns list of schemas."""
        return self._new_registry.get_schemas()
    
    def get_all_tool_schemas(self):
        """New-style: Returns JSON-serializable schemas."""
        return self._new_registry.get_all_tool_schemas()
    
    def register_langchain_tools(self, tools) -> None:
        """Bulk registers a list of LangChain tools."""
        self._new_registry.register_langchain_tools(tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._new_registry
    
    def __len__(self) -> int:
        return len(self._new_registry)


# For simple drop-in replacement
ToolRegistry = LegacyToolRegistry
