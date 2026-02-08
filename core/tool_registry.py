"""
Tool Registry - Centralized Tool Management with Hot-Loading

Epic 1, Story 1.4: Provides a centralized registry for discovering and managing 
tools with automatic JSON schema generation for LLM function calling.

Supports:
1. Functions decorated with @tool_schema
2. Dynamic hot-loading of new tools from tools/custom/active/
3. Callbacks to notify when toolset expands
"""

import os
import sys
import json
import inspect
import importlib.util
import threading
from typing import Any, Callable, Dict, List, Optional, get_type_hints, Literal
from functools import wraps
from dataclasses import dataclass, field, asdict

import logging

logger = logging.getLogger(__name__)


class ToolDefinitionError(Exception):
    """Raised when a tool function lacks required type hints or docstring."""
    pass


# =============================================================================
# Tool Result - Standardized Output Format
# =============================================================================

@dataclass
class ToolResult:
    """
    Standardized output format for all tools.
    
    Ensures consistent tool outputs that L.O.V.E. can 
    programmatically verify for success/failure.
    """
    status: Literal["success", "error"]
    data: Any = None
    observation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation for LLM consumption."""
        return self.observation


def wrap_tool_output(func: Callable) -> Callable:
    """
    Decorator that ensures tool functions return ToolResult.
    
    Wraps any tool function to standardize its output format.
    If the function raises an exception, it's caught and returned as an error ToolResult.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
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


# =============================================================================
# Schema Utilities
# =============================================================================

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
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
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
        
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_params_section = True
            continue
        
        if stripped.lower().startswith("returns:") or stripped.lower().startswith("raises:"):
            in_params_section = False
            continue
        
        if in_params_section:
            if ":" in stripped and not stripped.startswith(" "):
                parts = stripped.split(":", 1)
                param_name = parts[0].strip()
                if "(" in param_name:
                    param_name = param_name.split("(")[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                params[param_name] = param_desc
                current_param = param_name
            elif current_param and stripped:
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
    sig = inspect.signature(func)
    
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    
    docstring = func.__doc__
    if not docstring or not docstring.strip():
        raise ToolDefinitionError(
            f"Function '{func.__name__}' must have a docstring describing its purpose."
        )
    
    excluded_params = {"self", "cls", "kwargs", "args", "engine", "state"}
    params = {
        name: param
        for name, param in sig.parameters.items()
        if name not in excluded_params and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        )
    }
    
    missing_hints = [name for name in params if name not in hints]
    if missing_hints:
        raise ToolDefinitionError(
            f"Function '{func.__name__}' is missing type hints for parameters: {missing_hints}"
        )
    
    doc_info = _parse_docstring(docstring)
    
    properties = {}
    required = []
    
    for name, param in params.items():
        param_type = hints.get(name, str)
        param_schema = _python_type_to_json_schema(param_type)
        
        if name in doc_info["params"]:
            param_schema["description"] = doc_info["params"][name]
        
        properties[name] = param_schema
        
        if param.default is inspect.Parameter.empty:
            required.append(name)
    
    schema = {
        "name": func.__name__,
        "description": doc_info["description"],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    
    func.__tool_schema__ = schema
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__tool_schema__ = schema
    return wrapper


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    A centralized registry for discovering and managing tools with automatic
    JSON schema generation for LLM function calling.
    
    Features:
    - Register tools with @tool_schema decorator
    - Hot-load new tools from tools/custom/active/
    - Callbacks when new tools are added
    - Polling-based file watching (more reliable than watchdog)
    """
    
    ACTIVE_TOOLS_DIR = "tools/custom/active"
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._custom_tools: set = set()  # Track which tools came from custom/
        self._on_tool_added_callbacks: List[Callable[[str], None]] = []
        self._watcher_thread: Optional[threading.Thread] = None
        self._watching = False
        self._known_files: set = set()
        
        # Get project root
        self._project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self._active_dir = os.path.join(self._project_root, self.ACTIVE_TOOLS_DIR)
    
    def register(self, func: Callable, name: Optional[str] = None) -> None:
        """
        Registers a tool with automatic schema extraction.
        
        Args:
            func: The function to register
            name: Optional override for the tool name
        
        Raises:
            ToolDefinitionError: If schema cannot be extracted
        """
        tool_name = name or getattr(func, "name", None) or func.__name__
        
        schema = None
        
        # Check for @tool_schema decorator
        if hasattr(func, "__tool_schema__"):
            schema = func.__tool_schema__
        
        # Check for description attribute
        elif hasattr(func, "description"):
            schema = {
                "name": tool_name,
                "description": func.description,
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        
        # Fall back to docstring-based extraction
        else:
            if not func.__doc__:
                raise ToolDefinitionError(
                    f"Cannot register '{tool_name}': No schema found and no docstring available."
                )
            
            try:
                decorated = tool_schema(func)
                schema = decorated.__tool_schema__
                func = decorated
            except ToolDefinitionError as e:
                raise ToolDefinitionError(f"Cannot register '{tool_name}': {e}")
        
        if tool_name in self._tools:
            existing_tool = self._tools[tool_name].get("func")
            if existing_tool is func:
                return
            logger.warning(f"Tool '{tool_name}' is already registered. Overwriting.")
        
        # Ensure schema name matches registration name (critical for LLM calling)
        if schema and schema.get("name") != tool_name:
            logger.info(f"Renaming schema for tool '{schema.get('name')}' to '{tool_name}'")
            schema["name"] = tool_name
        
        self._tools[tool_name] = {
            "func": func,
            "schema": schema
        }
        
        # Log to console for visibility
        print(f"\n🔧 Tool Registered: {tool_name}")
        if schema:
            if schema.get("description"):
                desc = schema["description"].split("\n")[0]
                print(f"   📝 Description: {desc}")
            params = schema.get("parameters", {})
            properties = params.get("properties", {})
            required = params.get("required", [])
            if properties:
                print(f"   📋 Parameters:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = "* " if param_name in required else "  "
                    if param_desc:
                        print(f"      {req_marker}{param_name} ({param_type}): {param_desc}")
                    else:
                        print(f"      {req_marker}{param_name} ({param_type})")
            else:
                print(f"   📋 Parameters: None")
            
        # Notify listeners
        self._notify_tool_added(tool_name)
    
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
        """Returns a list of JSON schemas in OpenAI function calling format."""
        return [tool["schema"] for tool in self._tools.values()]
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Returns the schema for a specific tool."""
        if name in self._tools:
            return self._tools[name]["schema"]
        return None
    
    def list_tools(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self._tools.keys())
    
    def get_formatted_tool_metadata(self) -> str:
        """
        Returns a formatted string of all tool metadata, suitable for
        injection into an LLM prompt.
        """
        if not self._tools:
            return "No tools are available."
        
        output = "You have access to the following tools:\n\n"
        
        for name, data in self._tools.items():
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
        
        return output
    
    # =========================================================================
    # Hot-Loading (Story 1.4)
    # =========================================================================
    
    def load_active_tools(self) -> List[str]:
        """
        Load all tools from the active directory.
        
        Returns:
            List of loaded tool names
        """
        if not os.path.exists(self._active_dir):
            return []
        
        loaded_tools = []
        
        for filename in os.listdir(self._active_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                tool_name = self._load_tool_module(
                    os.path.join(self._active_dir, filename)
                )
                if tool_name:
                    loaded_tools.append(tool_name)
        
        return loaded_tools
    
    def _load_tool_module(self, file_path: str) -> Optional[str]:
        """
        Dynamically load a tool module from a file.
        
        Returns:
            Tool name if loaded successfully, None otherwise
        """
        module_name = os.path.basename(file_path)[:-3]  # Remove .py
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"tools.custom.active.{module_name}",
                file_path
            )
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {file_path}")
                return None
            
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
                        self._custom_tools.add(attr.__tool_schema__["name"])
                        logger.info(f"🔧 Loaded custom tool: {attr_name} from {os.path.basename(file_path)}")
                        return attr.__tool_schema__["name"]
                    except ToolDefinitionError as e:
                        logger.warning(f"Failed to register {attr_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load module {file_path}: {e}")
        
        return None
    
    def start_watching(self, interval: float = 2.0):
        """
        Start watching the active directory for new tools.
        
        Uses polling instead of watchdog for reliability.
        
        Args:
            interval: Seconds between checks
        """
        if self._watching:
            return
        
        self._watching = True
        self._known_files = set()
        
        # Initialize known files
        if os.path.exists(self._active_dir):
            for f in os.listdir(self._active_dir):
                if f.endswith(".py"):
                    self._known_files.add(f)
        
        def watch_loop():
            while self._watching:
                try:
                    if os.path.exists(self._active_dir):
                        current_files = {
                            f for f in os.listdir(self._active_dir) 
                            if f.endswith(".py") and not f.startswith("_")
                        }
                        
                        new_files = current_files - self._known_files
                        for new_file in new_files:
                            file_path = os.path.join(self._active_dir, new_file)
                            tool_name = self._load_tool_module(file_path)
                            if tool_name:
                                self._notify_tool_added(tool_name)
                        
                        self._known_files = current_files
                        
                except Exception as e:
                    logger.error(f"Error in file watcher: {e}")
                
                threading.Event().wait(interval)
        
        self._watcher_thread = threading.Thread(target=watch_loop, daemon=True)
        self._watcher_thread.start()
        logger.info("🔍 Started watching for new tools")
    
    def stop_watching(self):
        """Stop watching for new tools."""
        self._watching = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
            self._watcher_thread = None
        logger.info("⏹️ Stopped watching for new tools")
    
    def on_tool_added(self, callback: Callable[[str], None]):
        """
        Register a callback for when a new tool is added.
        
        Args:
            callback: Function that takes tool name as argument
        """
        self._on_tool_added_callbacks.append(callback)
    
    def _notify_tool_added(self, tool_name: str):
        """Notify all callbacks that a new tool was added."""
        for callback in self._on_tool_added_callbacks:
            try:
                callback(tool_name)
            except Exception as e:
                logger.error(f"Error in tool-added callback: {e}")
    
    def refresh(self) -> Dict[str, Any]:
        """
        Refresh the registry by reloading custom tools.
        
        Returns:
            {"loaded": List[str], "total": int}
        """
        # Clear existing custom tools
        for name in list(self._custom_tools):
            if name in self._tools:
                del self._tools[name]
        self._custom_tools.clear()
        
        # Reload
        loaded = self.load_active_tools()
        
        result = {
            "loaded": loaded,
            "total": len(self._tools)
        }
        
        logger.info(f"🔄 Registry refreshed: {len(loaded)} custom tools loaded, {result['total']} total")
        return result
    
    def unregister(self, name: str) -> bool:
        """
        Removes a tool from the registry.
        
        Args:
            name: Name of tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            self._custom_tools.discard(name)
            return True
        return False
    
    def get_custom_tools(self) -> List[str]:
        """Returns a list of tool names that were dynamically loaded."""
        return list(self._custom_tools)
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools


# =============================================================================
# Global Registry
# =============================================================================

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
