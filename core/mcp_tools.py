# core/mcp_tools.py
"""
MCP Tool Integration Module.

Exposes MCP server tools (like GitHub) as callable tool functions that can be
registered with the ToolRegistry and invoked from the REPL or LLM agents.
"""

import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import tool

import core.logging
import core.shared_state as shared_state


class MCPToolWrapper:
    """
    Wraps an MCP server tool call to make it callable as a standard function.
    
    Handles:
    - Starting the MCP server if not running
    - Sending JSON-RPC tool calls
    - Waiting for and parsing responses
    - Error handling and retries
    """
    
    def __init__(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        mcp_manager=None,
        timeout: int = 30
    ):
        self.server_name = server_name
        self.tool_name = tool_name
        self.description = description
        self._mcp_manager = mcp_manager
        self.timeout = timeout
    
    @property
    def mcp_manager(self):
        """Get MCP manager from shared_state if not provided directly."""
        if self._mcp_manager:
            return self._mcp_manager
        return getattr(shared_state, 'mcp_manager', None)
    
    def _ensure_server_running(self) -> bool:
        """Ensures the MCP server is running, starts it if needed."""
        manager = self.mcp_manager
        if not manager:
            core.logging.log_event(
                f"MCP manager not available for {self.server_name}",
                "ERROR"
            )
            return False
        
        running_servers = manager.list_running_servers()
        server_running = any(
            s.get('name') == self.server_name for s in running_servers
        )
        
        if not server_running:
            core.logging.log_event(
                f"Starting MCP server '{self.server_name}' for tool call",
                "INFO"
            )
            result = manager.start_server(self.server_name)
            if "Error" in result:
                core.logging.log_event(f"Failed to start MCP server: {result}", "ERROR")
                return False
        
        return True
    
    async def __call__(self, **kwargs) -> str:
        """
        Invokes the MCP tool with the provided arguments.
        
        Args:
            **kwargs: Arguments to pass to the MCP tool
            
        Returns:
            String result from the MCP tool execution
        """
        manager = self.mcp_manager
        if not manager:
            return f"Error: MCP manager not available"
        
        # Ensure server is running
        if not self._ensure_server_running():
            return f"Error: Could not start MCP server '{self.server_name}'"
        
        try:
            # Call the tool
            request_id = manager.call_tool(
                self.server_name,
                self.tool_name,
                kwargs
            )
            
            # Wait for response
            response = manager.get_response(
                self.server_name,
                request_id,
                timeout=self.timeout
            )
            
            # Parse response
            if "error" in response:
                error_msg = response["error"].get("message", str(response["error"]))
                return f"Error: {error_msg}"
            
            result = response.get("result", {})
            
            # Handle content array format (MCP standard)
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list):
                    # Extract text from content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            texts.append(block["text"])
                    return "\n".join(texts) if texts else str(result)
                return str(content)
            
            return str(result)
            
        except ValueError as e:
            return f"Error: {e}"
        except IOError as e:
            return f"Error communicating with MCP server: {e}"
        except Exception as e:
            core.logging.log_event(
                f"Unexpected error invoking MCP tool {self.tool_name}: {e}",
                "ERROR"
            )
            return f"Error: {e}"


def create_mcp_tool(
    server_name: str,
    tool_name: str,
    description: str,
    mcp_manager=None
) -> Callable:
    """
    Factory function to create a LangChain-compatible tool from an MCP tool.
    
    Args:
        server_name: Name of the MCP server (e.g., "github")
        tool_name: Name of the tool on the server (e.g., "repos.search_repositories")
        description: Description of what the tool does
        mcp_manager: Optional MCPManager instance (uses shared_state if not provided)
        
    Returns:
        A decorated async function that can be registered with ToolRegistry
    """
    # Create the wrapper
    wrapper = MCPToolWrapper(
        server_name=server_name,
        tool_name=tool_name,
        description=description,
        mcp_manager=mcp_manager
    )
    
    # Create registration name (e.g., "github.search_repositories")
    full_name = f"{server_name}.{tool_name.replace('.', '_')}"
    
    # Create dynamic input schema based on tool name
    # For now, use a generic schema with kwargs
    class MCPToolInput(BaseModel):
        query: str = Field(default="", description="Query or primary argument for the tool")
        kwargs: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional arguments as key-value pairs"
        )
    
    @tool(full_name, args_schema=MCPToolInput)
    async def mcp_tool_func(query: str = "", kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Invokes an MCP server tool."""
        # Merge query into kwargs if provided
        call_kwargs = kwargs or {}
        if query:
            call_kwargs["query"] = query
        return await wrapper(**call_kwargs)
    
    # Override the docstring with the actual description
    mcp_tool_func.__doc__ = description
    mcp_tool_func.description = description
    
    return mcp_tool_func


def register_mcp_tools(tool_registry, mcp_manager=None) -> List[str]:
    """
    Registers all available MCP server tools with the tool registry.
    
    Iterates through all configured MCP servers and their tools,
    creating wrapper functions for each one.
    
    Args:
        tool_registry: The ToolRegistry instance to register tools with
        mcp_manager: Optional MCPManager instance (uses shared_state if not provided)
        
    Returns:
        List of registered tool names
    """
    manager = mcp_manager or getattr(shared_state, 'mcp_manager', None)
    if not manager:
        core.logging.log_event(
            "MCP manager not available, skipping MCP tool registration",
            "WARNING"
        )
        return []
    
    registered_tools = []
    
    # Get server configs
    server_configs = getattr(manager, 'server_configs', {})
    
    for server_name, config in server_configs.items():
        tools = config.get("tools", {})
        
        for tool_name, description in tools.items():
            try:
                # Create and register the tool
                mcp_tool = create_mcp_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    description=description,
                    mcp_manager=manager
                )
                
                tool_registry.register(mcp_tool)
                full_name = f"{server_name}.{tool_name.replace('.', '_')}"
                registered_tools.append(full_name)
                
                core.logging.log_event(
                    f"Registered MCP tool: {full_name}",
                    "DEBUG"
                )
                
            except Exception as e:
                core.logging.log_event(
                    f"Failed to register MCP tool {server_name}.{tool_name}: {e}",
                    "WARNING"
                )
    
    if registered_tools:
        core.logging.log_event(
            f"Registered {len(registered_tools)} MCP tools: {', '.join(registered_tools)}",
            "INFO"
        )
    
    return registered_tools


def get_mcp_tools_summary(mcp_manager=None) -> str:
    """
    Returns a formatted summary of available MCP tools.
    
    Args:
        mcp_manager: Optional MCPManager instance
        
    Returns:
        Formatted string listing all MCP tools by server
    """
    manager = mcp_manager or getattr(shared_state, 'mcp_manager', None)
    if not manager:
        return "No MCP manager available."
    
    server_configs = getattr(manager, 'server_configs', {})
    
    if not server_configs:
        return "No MCP servers configured."
    
    lines = ["## MCP Server Tools\n"]
    
    for server_name, config in server_configs.items():
        tools = config.get("tools", {})
        required_env = config.get("requires_env", [])
        
        lines.append(f"### {server_name}")
        
        if required_env:
            lines.append(f"  Required ENV: {', '.join(required_env)}")
        
        if tools:
            for tool_name, description in tools.items():
                full_name = f"{server_name}.{tool_name.replace('.', '_')}"
                lines.append(f"  - **{full_name}**: {description}")
        else:
            lines.append("  No tools defined")
        
        lines.append("")
    
    return "\n".join(lines)
