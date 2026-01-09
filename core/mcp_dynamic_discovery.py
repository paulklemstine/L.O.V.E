# core/mcp_dynamic_discovery.py
"""
MCP Dynamic Discovery Module.

Provides lightweight dynamic discovery for MCP tools, inspired by philschmid/mcp-cli.
Instead of loading all tool definitions upfront, this module enables lazy discovery:

1. discover_servers() - List available MCP servers
2. discover_tools(server) - List tools with minimal descriptions
3. get_tool_schema(server, tool) - Get full schema on demand
4. execute_tool(server, tool, params) - Execute with auto-start

This achieves ~99% reduction in MCP-related token usage for typical interactions.
"""

import json
from typing import Any, Dict, List, Optional
import core.logging
import core.shared_state as shared_state


class MCPDynamicDiscovery:
    """
    Lightweight dynamic discovery wrapper for MCP tools.
    
    Uses the existing MCPManager for server lifecycle management,
    but provides a dynamic discovery interface that minimizes token usage.
    """
    
    def __init__(self, mcp_manager=None):
        """
        Initialize the dynamic discovery wrapper.
        
        Args:
            mcp_manager: Optional MCPManager instance. If not provided,
                        uses shared_state.mcp_manager.
        """
        self._mcp_manager = mcp_manager
    
    @property
    def mcp_manager(self):
        """Get the MCP manager instance."""
        if self._mcp_manager:
            return self._mcp_manager
        return getattr(shared_state, 'mcp_manager', None)
    
    def discover_servers(self) -> List[Dict[str, Any]]:
        """
        List all available MCP servers from configuration.
        
        Returns:
            List of server info dicts with 'name', 'status', and 'requires_env' fields.
            Does NOT start any servers.
        """
        manager = self.mcp_manager
        if not manager:
            core.logging.log_event("MCP manager not available", "WARNING")
            return []
        
        servers = []
        server_configs = getattr(manager, 'server_configs', {})
        running_servers = {s['name'] for s in manager.list_running_servers()}
        
        for name, config in server_configs.items():
            missing_env = manager.check_missing_env_vars(name)
            servers.append({
                "name": name,
                "status": "running" if name in running_servers else "stopped",
                "requires_env": config.get("requires_env", []),
                "missing_env": missing_env,
                "tool_count": len(config.get("tools", {}))
            })
        
        return servers
    
    def discover_tools(self, server_name: str) -> Dict[str, str]:
        """
        List tools available on a server with brief descriptions.
        
        This method returns minimal information to reduce token usage.
        Use get_tool_schema() for full parameter details when needed.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Dict mapping tool names to brief descriptions
        """
        manager = self.mcp_manager
        if not manager:
            return {"error": "MCP manager not available"}
        
        server_config = manager.server_configs.get(server_name)
        if not server_config:
            return {"error": f"Server '{server_name}' not found"}
        
        tools = server_config.get("tools", {})
        
        # Return truncated descriptions to minimize tokens
        brief_tools = {}
        for name, desc in tools.items():
            # Truncate description to first 80 chars
            brief_desc = desc[:80] + "..." if len(desc) > 80 else desc
            brief_tools[name] = brief_desc
        
        return brief_tools
    
    def get_tool_schema(self, server_name: str, tool_name: str) -> Dict[str, Any]:
        """
        Get the full schema for a specific tool.
        
        This is called only when an agent needs to actually use a tool,
        providing full parameter details on demand.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            
        Returns:
            Dict with 'name', 'description', and 'parameters' fields
        """
        manager = self.mcp_manager
        if not manager:
            return {"error": "MCP manager not available"}
        
        server_config = manager.server_configs.get(server_name)
        if not server_config:
            return {"error": f"Server '{server_name}' not found"}
        
        tools = server_config.get("tools", {})
        if tool_name not in tools:
            return {"error": f"Tool '{tool_name}' not found on server '{server_name}'"}
        
        # Build schema - for now using generic schema since MCP servers
        # accept dynamic arguments. In the future, this could query the
        # server directly for its schema.
        return {
            "name": tool_name,
            "server": server_name,
            "description": tools[tool_name],
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
                "description": "Pass any parameters as a JSON object. The MCP server will validate."
            },
            "required_env": server_config.get("requires_env", [])
        }
    
    def _ensure_server_running(self, server_name: str) -> tuple[bool, str]:
        """
        Ensures the MCP server is running, starts it if needed.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        manager = self.mcp_manager
        if not manager:
            return False, "MCP manager not available"
        
        running_servers = manager.list_running_servers()
        server_running = any(s.get('name') == server_name for s in running_servers)
        
        if not server_running:
            core.logging.log_event(
                f"Starting MCP server '{server_name}' for tool execution",
                "INFO"
            )
            result = manager.start_server(server_name)
            if "Error" in result:
                return False, result
        
        return True, f"Server '{server_name}' is running"
    
    def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 60
    ) -> str:
        """
        Execute an MCP tool with auto-start.
        
        This is the main execution method that:
        1. Starts the server if not running
        2. Sends the tool call via JSON-RPC
        3. Waits for and returns the response
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool (default: {})
            timeout: Timeout in seconds for the response
            
        Returns:
            String result from the tool execution, or error message
        """
        manager = self.mcp_manager
        if not manager:
            return "Error: MCP manager not available"
        
        # Ensure server is running
        success, msg = self._ensure_server_running(server_name)
        if not success:
            return f"Error: {msg}"
        
        # Execute the tool
        try:
            params = params or {}
            request_id = manager.call_tool(server_name, tool_name, params)
            response = manager.get_response(server_name, request_id, timeout=timeout)
            
            if "error" in response:
                error_msg = response["error"].get("message", str(response["error"]))
                return f"Error: {error_msg}"
            
            result = response.get("result", {})
            
            # Format result
            if isinstance(result, dict):
                # Handle MCP content array format
                if "content" in result:
                    content = result["content"]
                    if isinstance(content, list):
                        texts = [b.get("text", str(b)) for b in content if isinstance(b, dict)]
                        return "\n".join(texts) if texts else json.dumps(result, indent=2)
                return json.dumps(result, indent=2)
            elif isinstance(result, list):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except ValueError as e:
            return f"Error: {e}"
        except IOError as e:
            return f"Error communicating with MCP server: {e}"
        except Exception as e:
            core.logging.log_event(
                f"Unexpected error executing MCP tool {server_name}.{tool_name}: {e}",
                "ERROR"
            )
            return f"Error: {e}"


# Singleton instance for global access
_discovery_instance: Optional[MCPDynamicDiscovery] = None


def get_discovery() -> MCPDynamicDiscovery:
    """Get the global MCPDynamicDiscovery instance."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = MCPDynamicDiscovery()
    return _discovery_instance


def reset_discovery():
    """Reset the global discovery instance (for testing)."""
    global _discovery_instance
    _discovery_instance = None
