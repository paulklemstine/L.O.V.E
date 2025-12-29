"""
Secure Executor for running tools in a controlled environment.
Migrated from tools_legacy.py as part of the legacy purge (Story 1.4).

This module provides secure, async-compatible tool execution with proper
error handling and registry integration.
"""
import asyncio
from typing import Any, Dict, Optional
from core.tool_registry import ToolRegistry
import logging


class SecureExecutor:
    """
    A secure environment for running tool code.
    
    This executor provides:
    - Async-compatible tool execution
    - Registry-based tool lookup
    - Proper error handling and logging
    - Support for both sync and async tools
    """
    
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize the SecureExecutor.
        
        Args:
            tool_registry: Optional ToolRegistry instance. If not provided,
                          the global registry will be used.
        """
        self._tool_registry = tool_registry
    
    @property
    def tool_registry(self) -> ToolRegistry:
        """Gets the tool registry, falling back to global if not set."""
        if self._tool_registry is None:
            from core.tool_registry import get_global_registry
            return get_global_registry()
        return self._tool_registry
    
    async def execute(self, tool_name: str, tool_registry: Optional[ToolRegistry] = None, **kwargs: Any) -> str:
        """
        Executes a given tool from the registry asynchronously.
        
        Args:
            tool_name: Name of the tool to execute
            tool_registry: Optional registry override (for backward compatibility)
            **kwargs: Arguments to pass to the tool
        
        Returns:
            String result of tool execution or error message
        """
        registry = tool_registry or self.tool_registry
        
        try:
            tool = registry.get_tool(tool_name)
            
            # Handle different tool types
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(kwargs)
            elif hasattr(tool, "invoke"):
                result = tool.invoke(kwargs)
            elif asyncio.iscoroutinefunction(tool):
                result = await tool(**kwargs)
            else:
                result = tool(**kwargs)
            
            return str(result)
        
        except KeyError:
            error_msg = f"Error: Tool '{tool_name}' not found in registry."
            logging.warning(error_msg)
            return error_msg
        
        except TypeError as e:
            error_msg = f"Error: Invalid arguments for tool '{tool_name}': {e}"
            logging.warning(error_msg)
            return error_msg
        
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}"
            logging.error(error_msg, exc_info=True)
            return error_msg
    
    def execute_sync(self, tool_name: str, tool_registry: Optional[ToolRegistry] = None, **kwargs: Any) -> str:
        """
        Synchronous wrapper for execute().
        
        Args:
            tool_name: Name of the tool to execute
            tool_registry: Optional registry override
            **kwargs: Arguments to pass to the tool
        
        Returns:
            String result of tool execution or error message
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.execute(tool_name, tool_registry, **kwargs))


# Convenience function for quick execution
async def execute_tool(tool_name: str, **kwargs: Any) -> str:
    """
    Convenience function to execute a tool from the global registry.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool
    
    Returns:
        String result of tool execution or error message
    """
    executor = SecureExecutor()
    return await executor.execute(tool_name, **kwargs)
