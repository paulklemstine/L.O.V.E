"""
Centralized Tool Registry with Automatic Schema Generation.

This module provides a decorator-based approach to tool registration that automatically
extracts JSON schemas (OpenAI function calling format) from Python function signatures
and docstrings.
"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, get_type_hints, Literal
from functools import wraps
from dataclasses import dataclass, field, asdict


class ToolDefinitionError(Exception):
    """Raised when a tool function lacks required type hints or docstring."""
    pass


# =============================================================================
# DeepAgent Protocol - Story 1.3: Tool Registry Standardization
# =============================================================================

@dataclass
class ToolResult:
    """
    Standardized output format for all tools.
    
    Story 1.3: Ensures consistent tool outputs that the DeepAgent can
    programmatically verify for success/failure.
    
    Attributes:
        status: Either "success" or "error"
        data: The actual payload/result from the tool
        observation: Human-readable text for the LLM to understand what happened
    """
    status: Literal["success", "error"]
    data: Any = None
    observation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation for LLM consumption."""
        return self.observation or str(self.data)


def wrap_tool_output(func: Callable) -> Callable:
    """
    Decorator that ensures tool functions return ToolResult.
    
    Story 1.3: Wraps any tool function to standardize its output format.
    If the function raises an exception, it's caught and returned as an error ToolResult.
    If the function returns a non-ToolResult, it's wrapped in a success ToolResult.
    
    Example:
        @wrap_tool_output
        def my_tool(x: int) -> str:
            return f"Result: {x}"
        
        result = my_tool(42)
        # result.status == "success"
        # result.data == "Result: 42"
        # result.observation == "Result: 42"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # If already a ToolResult, pass through
            if isinstance(result, ToolResult):
                return result
            
            # Wrap in ToolResult
            return ToolResult(
                status="success",
                data=result,
                observation=str(result) if result is not None else "Operation completed successfully."
            )
        except Exception as e:
            return ToolResult(
                status="error",
                data={"error_type": type(e).__name__, "error_message": str(e)},
                observation=f"Error: {type(e).__name__}: {str(e)}"
            )
    
    # Preserve any schema attributes
    if hasattr(func, "__tool_schema__"):
        wrapper.__tool_schema__ = func.__tool_schema__
    
    return wrapper


async def wrap_async_tool_output(func: Callable) -> Callable:
    """
    Async version of wrap_tool_output for async tool functions.
    
    Story 1.3: Same standardization for async tools.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            
            if isinstance(result, ToolResult):
                return result
            
            return ToolResult(
                status="success",
                data=result,
                observation=str(result) if result is not None else "Operation completed successfully."
            )
        except Exception as e:
            return ToolResult(
                status="error",
                data={"error_type": type(e).__name__, "error_message": str(e)},
                observation=f"Error: {type(e).__name__}: {str(e)}"
            )
    
    if hasattr(func, "__tool_schema__"):
        wrapper.__tool_schema__ = func.__tool_schema__
    
    return wrapper


def _python_type_to_json_schema(py_type: type) -> Dict[str, Any]:
    """Converts a Python type to JSON Schema type definition."""
    type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"},
    }
    
    # Handle Optional types (Union[X, None])
    origin = getattr(py_type, "__origin__", None)
    if origin is type(None):
        return {"type": "null"}
    
    # Handle List[X], Dict[X, Y], etc.
    if origin is list:
        args = getattr(py_type, "__args__", ())
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}
    
    if origin is dict:
        return {"type": "object"}
    
    # Handle Union types (including Optional)
    if hasattr(py_type, "__origin__") and str(py_type.__origin__) == "typing.Union":
        args = getattr(py_type, "__args__", ())
        # Filter out NoneType for Optional
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        # For complex unions, just use string
        return {"type": "string"}
    
    return type_mapping.get(py_type, {"type": "string"})


def _parse_docstring(docstring: str) -> Dict[str, str]:
    """
    Parses a docstring to extract description and parameter descriptions.
    
    Returns:
        Dict with 'description' and 'params' keys.
    """
    if not docstring:
        return {"description": "", "params": {}}
    
    lines = docstring.strip().split("\n")
    description_lines = []
    params = {}
    current_param = None
    in_params_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check for Args: or Parameters: section
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_params_section = True
            continue
        
        # Check for Returns: section (end of params)
        if stripped.lower().startswith("returns:") or stripped.lower().startswith("raises:"):
            in_params_section = False
            continue
        
        if in_params_section:
            # Parse parameter line: "param_name: description" or "param_name (type): description"
            if ":" in stripped and not stripped.startswith(" "):
                parts = stripped.split(":", 1)
                param_name = parts[0].strip()
                # Remove type annotation if present: "param (int)" -> "param"
                if "(" in param_name:
                    param_name = param_name.split("(")[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                params[param_name] = param_desc
                current_param = param_name
            elif current_param and stripped:
                # Continuation of previous param description
                params[current_param] += " " + stripped
        else:
            if stripped:
                description_lines.append(stripped)
    
    return {
        "description": " ".join(description_lines),
        "params": params
    }


def tool_schema(func: Callable) -> Callable:
    """
    Decorator that validates a function has proper type hints and docstring,
    then attaches schema metadata for registry extraction.
    
    Raises:
        ToolDefinitionError: If the function lacks type hints or docstring.
    
    Example:
        @tool_schema
        def calculate_sum(a: int, b: int) -> int:
            '''Calculates the sum of two integers.
            
            Args:
                a: First integer
                b: Second integer
            
            Returns:
                The sum of a and b
            '''
            return a + b
    """
    # Get function signature
    sig = inspect.signature(func)
    
    # Get type hints
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    
    # Validate docstring exists
    docstring = func.__doc__
    if not docstring or not docstring.strip():
        raise ToolDefinitionError(
            f"Function '{func.__name__}' must have a docstring describing its purpose."
        )
    
    # Get parameters (excluding 'self', 'cls', 'kwargs', 'engine', etc.)
    excluded_params = {"self", "cls", "kwargs", "args", "engine", "state"}
    params = {
        name: param
        for name, param in sig.parameters.items()
        if name not in excluded_params and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        )
    }
    
    # Validate type hints exist for all parameters
    missing_hints = [name for name in params if name not in hints]
    if missing_hints:
        raise ToolDefinitionError(
            f"Function '{func.__name__}' is missing type hints for parameters: {missing_hints}"
        )
    
    # Parse docstring
    doc_info = _parse_docstring(docstring)
    
    # Build JSON schema for parameters
    properties = {}
    required = []
    
    for name, param in params.items():
        param_type = hints.get(name, str)
        param_schema = _python_type_to_json_schema(param_type)
        
        # Add description from docstring
        if name in doc_info["params"]:
            param_schema["description"] = doc_info["params"][name]
        
        properties[name] = param_schema
        
        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    
    # Build complete schema
    schema = {
        "name": func.__name__,
        "description": doc_info["description"],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    
    # Attach schema to function
    func.__tool_schema__ = schema
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__tool_schema__ = schema
    return wrapper


class ToolRegistry:
    """
    A centralized registry for discovering and managing tools with automatic
    JSON schema generation for LLM function calling.
    
    This registry can work with:
    1. Functions decorated with @tool_schema
    2. LangChain @tool decorated functions
    3. Raw functions with manual metadata
    
    Example:
        registry = ToolRegistry()
        
        @tool_schema
        def my_tool(x: int) -> str:
            '''Does something useful.'''
            return str(x)
        
        registry.register(my_tool)
        
        # Get schemas for LLM binding
        schemas = registry.get_schemas()
        
        # Execute a tool
        tool = registry.get_tool("my_tool")
        result = tool(42)
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, func: Callable, name: Optional[str] = None) -> None:
        """
        Registers a tool with automatic schema extraction.
        
        Supports functions decorated with @tool_schema, LangChain @tool,
        or raw functions with __tool_schema__ attribute.
        
        Args:
            func: The function to register
            name: Optional override for the tool name
        
        Raises:
            ToolDefinitionError: If schema cannot be extracted
        """
        tool_name = name or getattr(func, "name", None) or func.__name__
        
        # Try to get schema from different sources
        schema = None
        
        # 1. Check for @tool_schema decorator
        if hasattr(func, "__tool_schema__"):
            schema = func.__tool_schema__
        
        # 2. Check for LangChain @tool decorator
        elif hasattr(func, "args_schema") and func.args_schema:
            # LangChain tool with Pydantic schema
            pydantic_schema = func.args_schema
            schema = {
                "name": tool_name,
                "description": getattr(func, "description", func.__doc__ or ""),
                "parameters": pydantic_schema.model_json_schema() if hasattr(pydantic_schema, "model_json_schema") else {}
            }
        
        # 3. Check for description attribute (LangChain BaseTool)
        elif hasattr(func, "description"):
            schema = {
                "name": tool_name,
                "description": func.description,
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        
        # 4. Fall back to docstring-based extraction
        else:
            if not func.__doc__:
                raise ToolDefinitionError(
                    f"Cannot register '{tool_name}': No schema found and no docstring available."
                )
            
            # Try to auto-generate schema
            try:
                decorated = tool_schema(func)
                schema = decorated.__tool_schema__
                func = decorated
            except ToolDefinitionError as e:
                raise ToolDefinitionError(
                    f"Cannot register '{tool_name}': {e}"
                )
        
        if tool_name in self._tools:
            print(f"Warning: Tool '{tool_name}' is already registered. Overwriting.")
        
        self._tools[tool_name] = {
            "func": func,
            "schema": schema
        }
    
    def get_tool(self, name: str) -> Callable:
        """
        Retrieves a tool's callable function by its name.
        
        Args:
            name: The name of the tool
        
        Returns:
            The callable function
        
        Raises:
            KeyError: If the tool is not found
        """
        if name not in self._tools:
            available = ", ".join(list(self._tools.keys())[:10])
            raise KeyError(
                f"Tool '{name}' not found in registry. Available tools: {available}..."
            )
        return self._tools[name]["func"]
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns a list of JSON schemas in OpenAI function calling format.
        
        Returns:
            List of schema dictionaries suitable for LLM tool binding
        """
        return [tool["schema"] for tool in self._tools.values()]
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Returns the schema for a specific tool.
        
        Args:
            name: The name of the tool
        
        Returns:
            The schema dictionary or None if not found
        """
        if name in self._tools:
            return self._tools[name]["schema"]
        return None
    
    def list_tools(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_names(self) -> List[str]:
        """Returns a list of all registered tool names (alias for list_tools)."""
        return self.list_tools()
    
    def get_formatted_tool_metadata(self) -> str:
        """
        Returns a formatted string of all tool metadata, suitable for
        injection into an LLM prompt.
        
        Epic 2: Story 2.2 - Enhanced to distinguish MCP tools for semantic selection.
        """
        if not self._tools:
            return "No tools are available."
        
        # Categorize tools into MCP and Native
        mcp_tools = {}
        native_tools = {}
        
        for name, data in self._tools.items():
            schema = data["schema"]
            description = schema.get("description", "No description available.")
            
            # Check if this is an MCP tool by looking for [MCP: prefix
            if description.startswith("[MCP:"):
                mcp_tools[name] = data
            else:
                native_tools[name] = data
        
        output = "You have access to the following tools:\n\n"
        
        # Native Tools Section
        if native_tools:
            output += "## Native Tools\n\n"
            for name, data in native_tools.items():
                schema = data["schema"]
                description = schema.get("description", "No description available.")
                params = schema.get("parameters", {})
                
                output += f"**Tool Name:** `{name}`\n"
                output += f"**Description:** {description}\n"
                if params.get("properties"):
                    output += f"**Parameters:**\n```json\n{json.dumps(params, indent=2)}\n```\n"
                else:
                    output += "**Parameters:** None\n"
                output += "---\n"
        
        # MCP Tools Section (Epic 2: Story 2.2)
        if mcp_tools:
            output += "\n## MCP External Tools\n\n"
            output += "These tools connect to external services via the Model Context Protocol.\n\n"
            
            for name, data in mcp_tools.items():
                schema = data["schema"]
                description = schema.get("description", "No description available.")
                params = schema.get("parameters", {})
                
                # Extract server name from [MCP:server_name] prefix
                server_name = ""
                if description.startswith("[MCP:"):
                    end_bracket = description.find("]")
                    if end_bracket > 5:
                        server_name = description[5:end_bracket]
                        description = description[end_bracket+1:].strip()
                
                output += f"**Tool Name:** `{name}`\n"
                if server_name:
                    output += f"**Server:** {server_name}\n"
                output += f"**Description:** {description}\n"
                if params.get("properties"):
                    output += f"**Parameters:**\n```json\n{json.dumps(params, indent=2)}\n```\n"
                else:
                    output += "**Parameters:** None\n"
                output += "---\n"
        
        return output
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns all tool schemas as JSON-serializable definitions.
        
        This is the canonical method for retrieving tool definitions
        for LLM binding and validation purposes.
        
        Returns:
            List of dictionaries containing tool name, description, 
            and parameters in OpenAI function calling format.
        """
        return [
            {
                "name": name,
                "description": data["schema"].get("description", ""),
                "parameters": data["schema"].get("parameters", {})
            }
            for name, data in self._tools.items()
        ]
    
    def register_langchain_tools(self, tools: List[Any]) -> None:
        """
        Bulk registers a list of LangChain tools.
        
        Args:
            tools: List of LangChain BaseTool instances
        """
        for tool in tools:
            try:
                self.register(tool)
            except ToolDefinitionError as e:
                print(f"Warning: Failed to register tool: {e}")

    # =========================================================================
    # Story 4.1: Dynamic Tool Loading (Tool Fabrication)
    # =========================================================================
    
    CUSTOM_TOOLS_DIR = "tools/custom"
    
    def load_custom_tools(self, custom_dir: str = None) -> List[str]:
        """
        Dynamically loads all Python modules from the custom tools directory.
        
        Story 4.1: Enables just-in-time tool fabrication by loading tools
        created by the agent at runtime.
        
        Args:
            custom_dir: Path to custom tools directory (defaults to tools/custom)
            
        Returns:
            List of loaded tool names
        """
        import os
        import sys
        import importlib.util
        
        if custom_dir is None:
            custom_dir = self.CUSTOM_TOOLS_DIR
        
        # Get absolute path
        if not os.path.isabs(custom_dir):
            # Assume relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            custom_dir = os.path.join(project_root, custom_dir)
        
        if not os.path.exists(custom_dir):
            return []
        
        loaded_tools = []
        
        for filename in os.listdir(custom_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]  # Remove .py extension
                file_path = os.path.join(custom_dir, filename)
                
                try:
                    # Load the module dynamically
                    spec = importlib.util.spec_from_file_location(
                        f"tools.custom.{module_name}",
                        file_path
                    )
                    if spec is None or spec.loader is None:
                        print(f"Warning: Could not load spec for {file_path}")
                        continue
                    
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)
                    
                    # Find functions with __tool_schema__ attribute
                    for attr_name in dir(module):
                        if attr_name.startswith("_"):
                            continue
                        
                        attr = getattr(module, attr_name)
                        if callable(attr) and hasattr(attr, "__tool_schema__"):
                            try:
                                self.register(attr)
                                loaded_tools.append(attr.__tool_schema__["name"])
                                print(f"🔧 Loaded custom tool: {attr_name} from {filename}")
                            except ToolDefinitionError as e:
                                print(f"Warning: Failed to register {attr_name}: {e}")
                    
                except Exception as e:
                    print(f"Warning: Failed to load module {file_path}: {e}")
        
        return loaded_tools
    
    def refresh(self) -> Dict[str, Any]:
        """
        Refreshes the tool registry by reloading custom tools.
        
        Story 4.1: Called after agent writes a new tool to tools/custom/
        to make it immediately available in the same execution trace.
        
        Returns:
            {
                "loaded": List[str],     # Names of newly loaded tools
                "failed": List[str],     # Names of tools that failed to load
                "total": int             # Total tools in registry
            }
        """
        # Track what we have before
        before_tools = set(self._tools.keys())
        
        # Clear custom tools before reloading
        custom_tools_to_remove = []
        for name, data in self._tools.items():
            # Check if this is a custom tool by checking module
            func = data.get("func")
            if func and hasattr(func, "__module__"):
                if "tools.custom" in str(func.__module__):
                    custom_tools_to_remove.append(name)
        
        for name in custom_tools_to_remove:
            self.unregister(name)
        
        # Reload custom tools
        loaded = self.load_custom_tools()
        
        # Determine what's new
        after_tools = set(self._tools.keys())
        new_tools = list(after_tools - before_tools)
        
        result = {
            "loaded": loaded,
            "new": new_tools,
            "total": len(self._tools)
        }
        
        print(f"🔄 Registry refreshed: {len(loaded)} custom tools loaded, {result['total']} total")
        
        return result
    
    def unregister(self, name: str) -> bool:
        """
        Removes a tool from the registry.
        
        Story 4.1: Allows removing tools that are no longer needed
        or need to be replaced.
        
        Args:
            name: Name of tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get_custom_tools(self) -> List[str]:
        """
        Returns a list of tool names that were dynamically loaded.
        
        Returns:
            List of custom tool names
        """
        custom_tools = []
        for name, data in self._tools.items():
            func = data.get("func")
            if func and hasattr(func, "__module__"):
                if "tools.custom" in str(func.__module__):
                    custom_tools.append(name)
        return custom_tools

    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global registry instance (optional, for convenience)
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Returns the global ToolRegistry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(func: Callable) -> Callable:
    """
    Decorator that both validates schema and registers to global registry.
    
    Example:
        @register_tool
        def my_tool(x: int) -> str:
            '''Does something.'''
            return str(x)
    """
    decorated = tool_schema(func)
    get_global_registry().register(decorated)
    return decorated
