# core/mcp_adapter.py
"""
MCP-to-LangChain Tool Adapter

Converts MCP (Model Context Protocol) tool definitions into LangChain BaseTool objects,
enabling agents to discover and invoke MCP tools through the standard LangChain interface.
"""

from typing import Any, Dict, List, Optional, Type
import json
import core.logging

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool, BaseTool


# JSON Schema type to Python type mapping
JSON_SCHEMA_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _json_schema_to_pydantic_field(name: str, schema: Dict[str, Any], required: bool = False) -> tuple:
    """
    Converts a JSON Schema property definition to a Pydantic Field tuple.
    
    Args:
        name: Field name
        schema: JSON Schema property definition
        required: Whether the field is required
        
    Returns:
        Tuple of (type, Field) for use with create_model
    """
    json_type = schema.get("type", "string")
    description = schema.get("description", f"Parameter: {name}")
    default = schema.get("default", ... if required else None)
    
    # Handle array types
    if json_type == "array":
        items_schema = schema.get("items", {})
        items_type = JSON_SCHEMA_TYPE_MAP.get(items_schema.get("type", "string"), Any)
        python_type = List[items_type]
    else:
        python_type = JSON_SCHEMA_TYPE_MAP.get(json_type, Any)
    
    # Create Field with appropriate default
    if required:
        field = Field(description=description)
        return (python_type, field)
    else:
        field = Field(default=default, description=description)
        return (Optional[python_type], field)


def _create_pydantic_model_from_schema(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model from a JSON Schema definition.
    
    Args:
        tool_name: Name to use for the model class
        schema: JSON Schema object with 'properties' and optionally 'required'
        
    Returns:
        A dynamically created Pydantic model class
    """
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    
    if not properties:
        # Empty schema - create a model with no fields
        return create_model(f"{tool_name}Input")
    
    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        is_required = prop_name in required_fields
        field_definitions[prop_name] = _json_schema_to_pydantic_field(
            prop_name, prop_schema, is_required
        )
    
    return create_model(f"{tool_name}Input", **field_definitions)


def _create_mcp_tool_wrapper(mcp_manager, server_name: str, tool_name: str):
    """
    Creates an async wrapper function that invokes an MCP tool.
    
    Args:
        mcp_manager: The MCPManager instance
        server_name: Name of the MCP server
        tool_name: Name of the tool on the MCP server
        
    Returns:
        An async function that can be used as a LangChain tool implementation
    """
    async def mcp_tool_executor(**kwargs) -> str:
        """Executes the MCP tool and returns the result as a string."""
        try:
            # Send the request to the MCP server
            request_id = mcp_manager.call_tool(server_name, tool_name, kwargs)
            
            # Wait for and retrieve the response
            response = mcp_manager.get_response(server_name, request_id, timeout=60)
            
            if "error" in response:
                error_msg = response["error"].get("message", "Unknown MCP error")
                core.logging.log_event(
                    f"MCP tool '{server_name}.{tool_name}' error: {error_msg}", "WARNING"
                )
                return f"Error calling MCP tool: {error_msg}"
            
            result = response.get("result", {})
            
            # Format result for agent consumption
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            elif isinstance(result, list):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except ValueError as e:
            return f"MCP server error: {e}"
        except IOError as e:
            return f"MCP communication error: {e}"
        except Exception as e:
            core.logging.log_event(
                f"Unexpected error in MCP tool '{server_name}.{tool_name}': {e}", "ERROR"
            )
            return f"Unexpected error: {e}"
    
    return mcp_tool_executor


def convert_mcp_to_langchain_tools(
    server_name: str, 
    mcp_manager,
    tool_definitions: Optional[Dict[str, str]] = None
) -> List[BaseTool]:
    """
    Converts MCP tool definitions into LangChain StructuredTool objects.
    
    This function takes tool definitions from an MCP server configuration and creates
    LangChain-compatible tools that can be used by agents. Each tool is prefixed with
    the server name to ensure uniqueness.
    
    Args:
        server_name: Name of the MCP server (e.g., "github")
        mcp_manager: The MCPManager instance for executing tool calls
        tool_definitions: Optional dict of {tool_name: description}. If not provided,
                         will attempt to get from server config.
                         
    Returns:
        List of LangChain BaseTool objects
        
    Example:
        ```python
        tools = convert_mcp_to_langchain_tools("github", mcp_manager)
        # Returns tools like: github_repos_search_repositories, github_issues_search_issues, etc.
        ```
    """
    langchain_tools = []
    
    # Get tool definitions from config if not provided
    if tool_definitions is None:
        server_config = mcp_manager.server_configs.get(server_name, {})
        tool_definitions = server_config.get("tools", {})
    
    if not tool_definitions:
        core.logging.log_event(
            f"No tool definitions found for MCP server '{server_name}'", "WARNING"
        )
        return []
    
    for mcp_tool_name, description in tool_definitions.items():
        # Create a unique tool name with server prefix
        # e.g., "repos.search_repositories" -> "github_repos_search_repositories"
        safe_tool_name = mcp_tool_name.replace(".", "_")
        langchain_name = f"{server_name}_{safe_tool_name}"
        
        # For now, we use a generic schema since MCP tools have dynamic arguments
        # The MCP server will validate the actual arguments
        generic_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or parameters for this tool"
                },
                "params": {
                    "type": "object",
                    "description": "Additional parameters as a JSON object"
                }
            },
            "required": []
        }
        
        try:
            # Create the Pydantic model for args_schema
            args_model = _create_pydantic_model_from_schema(langchain_name, generic_schema)
            
            # Create the execution wrapper
            executor = _create_mcp_tool_wrapper(mcp_manager, server_name, mcp_tool_name)
            
            # Create the LangChain StructuredTool
            tool = StructuredTool.from_function(
                func=executor,  # Will be treated as coroutine
                name=langchain_name,
                description=f"[MCP:{server_name}] {description}",
                args_schema=args_model,
                coroutine=executor  # Explicit async support
            )
            
            langchain_tools.append(tool)
            core.logging.log_event(
                f"Created LangChain tool '{langchain_name}' from MCP server '{server_name}'", "DEBUG"
            )
            
        except Exception as e:
            core.logging.log_event(
                f"Failed to create LangChain tool for '{mcp_tool_name}' from '{server_name}': {e}",
                "ERROR"
            )
    
    core.logging.log_event(
        f"Converted {len(langchain_tools)} MCP tools from server '{server_name}' to LangChain format",
        "INFO"
    )
    
    return langchain_tools


def get_all_mcp_langchain_tools(mcp_manager) -> List[BaseTool]:
    """
    Gets all available MCP tools from all running servers as LangChain tools.
    
    Args:
        mcp_manager: The MCPManager instance
        
    Returns:
        List of all LangChain BaseTool objects from all running MCP servers
    """
    all_tools = []
    
    running_servers = mcp_manager.list_running_servers()
    
    for server_info in running_servers:
        server_name = server_info["name"]
        tools = convert_mcp_to_langchain_tools(server_name, mcp_manager)
        all_tools.extend(tools)
    
    return all_tools
