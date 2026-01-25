"""
Tool Filesystem Adapter.

Exposes the L.O.V.E. ToolRegistry as a navigable filesystem.
Agents can browse tools, read schemas, and invoke tools by writing to special files.

Filesystem structure:
    /tools/
    ├── code_modifier/
    │   ├── schema.json          # Tool schema (JSON Schema format)
    │   ├── description.txt      # Human-readable description
    │   └── invoke               # Write arguments here to invoke tool
    ├── execute/
    │   ├── schema.json
    │   ├── description.txt
    │   └── invoke
    └── ...

Usage:
    # List available tools
    ls /tools
    
    # Read tool description
    cat /tools/execute/description.txt
    
    # Read tool schema
    cat /tools/execute/schema.json
    
    # Invoke a tool
    echo '{"command": "pwd"}' > /tools/execute/invoke
    cat /tools/execute/result  # Read the result
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from core.fuse.base import (
    FilesystemAdapter,
    FileAttributes,
    FileType,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
)

logger = logging.getLogger(__name__)


class ToolFilesystem(FilesystemAdapter):
    """
    Exposes the ToolRegistry as a virtual filesystem.
    
    Each tool becomes a directory containing:
    - schema.json: The tool's JSON schema
    - description.txt: Human-readable description
    - invoke: Write arguments here to execute the tool
    - result: Read the result of the last invocation
    """
    
    def __init__(self, tool_registry, mount_point: str = "/tools"):
        """
        Initialize with a ToolRegistry instance.
        
        Args:
            tool_registry: The L.O.V.E. ToolRegistry containing registered tools
            mount_point: Where to mount this filesystem
        """
        super().__init__(mount_point)
        self.tool_registry = tool_registry
        self._last_results: Dict[str, str] = {}  # tool_name -> last result
        self._pending_args: Dict[str, str] = {}  # tool_name -> pending arguments
    
    def _get_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        try:
            # Try different methods to get tool names
            if hasattr(self.tool_registry, 'list_tools'):
                tools = self.tool_registry.list_tools()
                if isinstance(tools, dict):
                    return list(tools.keys())
                return tools
            elif hasattr(self.tool_registry, 'get_tool_names'):
                return self.tool_registry.get_tool_names()
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting tool names: {e}")
            return []
    
    def _get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get the schema for a specific tool."""
        try:
            if hasattr(self.tool_registry, 'get_schema'):
                return self.tool_registry.get_schema(tool_name)
            elif hasattr(self.tool_registry, 'get_schemas'):
                schemas = self.tool_registry.get_schemas()
                for schema in schemas:
                    if schema.get("name") == tool_name or schema.get("function", {}).get("name") == tool_name:
                        return schema
            return None
        except Exception as e:
            logger.error(f"Error getting schema for {tool_name}: {e}")
            return None
    
    def _get_tool_callable(self, tool_name: str):
        """Get the callable for a tool."""
        try:
            if hasattr(self.tool_registry, 'get_tool'):
                return self.tool_registry.get_tool(tool_name)
            elif hasattr(self.tool_registry, 'list_tools'):
                tools = self.tool_registry.list_tools()
                if isinstance(tools, dict) and tool_name in tools:
                    tool_data = tools[tool_name]
                    if isinstance(tool_data, dict):
                        return tool_data.get("tool")
                    return tool_data
            return None
        except Exception as e:
            logger.error(f"Error getting callable for {tool_name}: {e}")
            return None
    
    def readdir(self, path: str) -> List[str]:
        """List directory contents."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            # Root: list all tools
            return self._get_tool_names()
        
        # Check if it's a tool directory
        parts = path.strip("/").split("/")
        tool_name = parts[0]
        
        if tool_name in self._get_tool_names():
            if len(parts) == 1:
                # Tool directory: list tool files
                return ["schema.json", "description.txt", "invoke", "result"]
            else:
                raise NotADirectoryError(f"Not a directory: {path}")
        
        raise FileNotFoundError(f"Directory not found: {path}")
    
    def read(self, path: str) -> str:
        """Read file contents."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) < 2:
            raise IsADirectoryError(f"Is a directory: {path}")
        
        tool_name = parts[0]
        filename = parts[1]
        
        if tool_name not in self._get_tool_names():
            raise FileNotFoundError(f"Tool not found: {tool_name}")
        
        schema = self._get_tool_schema(tool_name)
        
        if filename == "schema.json":
            if schema:
                return json.dumps(schema, indent=2)
            return json.dumps({"error": "Schema not available"})
        
        elif filename == "description.txt":
            if schema:
                # Extract description from schema
                if "description" in schema:
                    return schema["description"]
                elif "function" in schema and "description" in schema["function"]:
                    return schema["function"]["description"]
            return f"Tool: {tool_name}\nNo description available."
        
        elif filename == "result":
            # Return last invocation result
            return self._last_results.get(tool_name, "No result yet. Write to 'invoke' to execute the tool.")
        
        elif filename == "invoke":
            # Return pending arguments or usage info
            if tool_name in self._pending_args:
                return self._pending_args[tool_name]
            return self._get_usage_info(tool_name, schema)
        
        raise FileNotFoundError(f"File not found: {path}")
    
    def _get_usage_info(self, tool_name: str, schema: Optional[Dict]) -> str:
        """Generate usage information for a tool."""
        lines = [f"# Tool: {tool_name}", "", "## Usage"]
        lines.append("Write JSON arguments to this file to invoke the tool.")
        lines.append("Read 'result' file after invocation to get the output.")
        lines.append("")
        
        if schema:
            # Get parameters from schema
            params = {}
            if "parameters" in schema:
                params = schema["parameters"]
            elif "function" in schema and "parameters" in schema["function"]:
                params = schema["function"]["parameters"]
            
            if "properties" in params:
                lines.append("## Parameters")
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = param_name in params.get("required", [])
                    req_marker = " (required)" if required else ""
                    lines.append(f"- {param_name}: {param_type}{req_marker}")
                    if param_desc:
                        lines.append(f"    {param_desc}")
        
        lines.append("")
        lines.append("## Example")
        lines.append('echo \'{"param1": "value1"}\' > invoke')
        
        return "\n".join(lines)
    
    def write(self, path: str, content: str, append: bool = False) -> bool:
        """Write to a file, potentially invoking a tool."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) < 2:
            raise IsADirectoryError(f"Is a directory: {path}")
        
        tool_name = parts[0]
        filename = parts[1]
        
        if tool_name not in self._get_tool_names():
            raise FileNotFoundError(f"Tool not found: {tool_name}")
        
        if filename == "invoke":
            # Invoke the tool with the given arguments
            result = self._invoke_tool(tool_name, content.strip())
            self._last_results[tool_name] = result
            self._set_write_result(result)
            return True
        
        elif filename in ["schema.json", "description.txt"]:
            raise PermissionError(f"Cannot write to read-only file: {filename}")
        
        elif filename == "result":
            raise PermissionError("Cannot write to result file")
        
        raise FileNotFoundError(f"File not found: {path}")
    
    def _invoke_tool(self, tool_name: str, args_json: str) -> str:
        """Invoke a tool and return the result."""
        logger.info(f"ToolFilesystem invoking: {tool_name}")
        
        try:
            # Parse arguments
            if args_json:
                try:
                    args = json.loads(args_json)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON arguments: {e}"
            else:
                args = {}
            
            # Get the tool callable
            tool_callable = self._get_tool_callable(tool_name)
            if tool_callable is None:
                return f"Error: Tool '{tool_name}' not found or not callable"
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_callable):
                # Run async tool
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, tool_callable(**args))
                        result = future.result(timeout=60)
                else:
                    result = loop.run_until_complete(tool_callable(**args))
            elif hasattr(tool_callable, 'invoke'):
                # LangChain tool
                result = tool_callable.invoke(args)
            elif hasattr(tool_callable, '__call__'):
                result = tool_callable(**args)
            else:
                return f"Error: Tool '{tool_name}' is not callable"
            
            # Format result
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)
            
        except TypeError as e:
            return f"Error: Invalid arguments for '{tool_name}': {e}"
        except Exception as e:
            logger.exception(f"Tool invocation error: {e}")
            return f"Error: Tool execution failed: {e}"
    
    def getattr(self, path: str) -> FileAttributes:
        """Get file/directory attributes."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        parts = path.strip("/").split("/")
        tool_name = parts[0]
        
        if tool_name not in self._get_tool_names():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if len(parts) == 1:
            # Tool directory
            return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        filename = parts[1]
        
        if filename == "invoke":
            # Executable file (writing triggers action)
            return FileAttributes(mode=0o755, file_type=FileType.EXECUTABLE, size=0)
        
        elif filename in ["schema.json", "description.txt", "result"]:
            # Regular files
            content = self.read(path)
            return FileAttributes(
                mode=0o644,
                file_type=FileType.FILE,
                size=len(content.encode())
            )
        
        raise FileNotFoundError(f"Path not found: {path}")
