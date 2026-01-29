"""
Tests for Tool Registry - Phase 1

Epic 1, Story 1.4: Tests for Dynamic Hot-Loading
"""

import os
import sys
import pytest
import time
import tempfile
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_registry import (
    ToolRegistry,
    ToolDefinitionError,
    ToolResult,
    tool_schema,
    register_tool,
    get_global_registry,
    wrap_tool_output
)


class TestToolSchema:
    """Test the @tool_schema decorator."""
    
    def test_basic_decorator(self):
        """Should attach schema to function."""
        @tool_schema
        def my_tool(x: int, y: str) -> str:
            """Does something.
            
            Args:
                x: An integer
                y: A string
                
            Returns:
                Combined result
            """
            return f"{y}: {x}"
        
        assert hasattr(my_tool, "__tool_schema__")
        schema = my_tool.__tool_schema__
        
        assert schema["name"] == "my_tool"
        assert "Does something" in schema["description"]
        assert "x" in schema["parameters"]["properties"]
        assert "y" in schema["parameters"]["properties"]
    
    def test_missing_docstring(self):
        """Should raise error for missing docstring."""
        with pytest.raises(ToolDefinitionError):
            @tool_schema
            def no_doc(x: int) -> int:
                return x
    
    def test_missing_type_hints(self):
        """Should raise error for missing type hints."""
        with pytest.raises(ToolDefinitionError):
            @tool_schema
            def no_hints(x, y):
                """Has a docstring."""
                return x + y
    
    def test_optional_parameters(self):
        """Should mark parameters with defaults as optional."""
        @tool_schema
        def with_defaults(required: str, optional: int = 42) -> str:
            """Tool with defaults.
            
            Args:
                required: Required param
                optional: Optional param
            """
            return f"{required}: {optional}"
        
        schema = with_defaults.__tool_schema__
        
        assert "required" in schema["parameters"]["required"]
        assert "optional" not in schema["parameters"]["required"]


class TestToolRegistry:
    """Test the ToolRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return ToolRegistry()
    
    def test_register_decorated_function(self, registry):
        """Should register @tool_schema decorated function."""
        @tool_schema
        def test_tool(x: int) -> int:
            """Multi function.
            
            Args:
                x: Input
            """
            return x * 2
        
        registry.register(test_tool)
        
        assert "test_tool" in registry
        assert len(registry) == 1
    
    def test_register_raw_function_with_docstring(self, registry):
        """Should auto-generate schema from docstring."""
        def auto_schema(x: int) -> int:
            """Auto-schemaed tool.
            
            Args:
                x: Input value
            """
            return x * 2
        
        registry.register(auto_schema)
        
        assert "auto_schema" in registry
    
    def test_register_duplicate(self, registry):
        """Should warn on duplicate registration."""
        @tool_schema
        def dup_tool(x: int) -> int:
            """Tool A.
            
            Args:
                x: Input
            """
            return x
        
        @tool_schema  
        def dup_tool_v2(x: int) -> int:
            """Tool B.
            
            Args:
                x: Input
            """
            return x * 2
        
        registry.register(dup_tool)
        registry.register(dup_tool_v2, name="dup_tool")  # Same name, should warn
        
        assert len(registry) == 1
    
    def test_get_tool(self, registry):
        """Should retrieve tool by name."""
        @tool_schema
        def get_me(x: int) -> int:
            """Tool.
            
            Args:
                x: Input
            """
            return x
        
        registry.register(get_me)
        
        tool = registry.get_tool("get_me")
        assert tool(5) == 5
    
    def test_get_tool_not_found(self, registry):
        """Should raise KeyError for missing tool."""
        with pytest.raises(KeyError):
            registry.get_tool("nonexistent")
    
    def test_get_schemas(self, registry):
        """Should return all schemas."""
        @tool_schema
        def tool_a(x: int) -> int:
            """Tool A.
            
            Args:
                x: Input
            """
            return x
        
        @tool_schema
        def tool_b(y: str) -> str:
            """Tool B.
            
            Args:
                y: Input
            """
            return y
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        schemas = registry.get_schemas()
        
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "tool_a" in names
        assert "tool_b" in names
    
    def test_unregister(self, registry):
        """Should remove tool from registry."""
        @tool_schema
        def to_remove(x: int) -> int:
            """Tool.
            
            Args:
                x: Input
            """
            return x
        
        registry.register(to_remove)
        assert "to_remove" in registry
        
        result = registry.unregister("to_remove")
        
        assert result is True
        assert "to_remove" not in registry
    
    def test_formatted_metadata(self, registry):
        """Should format tools for LLM prompt."""
        @tool_schema
        def format_me(x: int) -> int:
            """A tool for formatting.
            
            Args:
                x: The input value
            """
            return x
        
        registry.register(format_me)
        
        metadata = registry.get_formatted_tool_metadata()
        
        assert "format_me" in metadata
        assert "A tool for formatting" in metadata


class TestHotLoading:
    """Test hot-loading functionality."""
    
    @pytest.fixture
    def registry_with_temp_dir(self, tmp_path):
        """Create registry with temp active directory."""
        registry = ToolRegistry()
        registry._active_dir = str(tmp_path / "active")
        os.makedirs(registry._active_dir, exist_ok=True)
        return registry
    
    def test_load_active_tools(self, registry_with_temp_dir):
        """Should load tools from active directory."""
        registry = registry_with_temp_dir
        
        # Create a tool file
        tool_code = '''
from core.tool_registry import tool_schema

@tool_schema
def hot_loaded_tool(x: int) -> int:
    """A hot-loaded tool.
    
    Args:
        x: Input value
    """
    return x * 3
'''
        with open(os.path.join(registry._active_dir, "hot_loaded_tool.py"), "w") as f:
            f.write(tool_code)
        
        loaded = registry.load_active_tools()
        
        assert len(loaded) >= 1
        assert "hot_loaded_tool" in registry
    
    def test_callback_on_tool_added(self, registry_with_temp_dir):
        """Should call callbacks when tool is added."""
        registry = registry_with_temp_dir
        
        added_tools = []
        registry.on_tool_added(lambda name: added_tools.append(name))
        
        # Simulate adding a tool
        registry._notify_tool_added("test_callback_tool")
        
        assert "test_callback_tool" in added_tools
    
    def test_refresh(self, registry_with_temp_dir):
        """Should reload custom tools on refresh."""
        registry = registry_with_temp_dir
        
        # Pre-populate with a tool
        @tool_schema
        def existing_tool(x: int) -> int:
            """Existing.
            
            Args:
                x: Input
            """
            return x
        
        registry.register(existing_tool)
        
        # Add a file to active
        tool_code = '''
from core.tool_registry import tool_schema

@tool_schema  
def refreshed_tool(x: int) -> int:
    """Refreshed.
    
    Args:
        x: Input
    """
    return x
'''
        with open(os.path.join(registry._active_dir, "refreshed_tool.py"), "w") as f:
            f.write(tool_code)
        
        result = registry.refresh()
        
        assert "refreshed_tool" in result["loaded"]
        assert result["total"] >= 1


class TestToolResult:
    """Test the ToolResult class."""
    
    def test_success_result(self):
        """Should create success result."""
        result = ToolResult(
            status="success",
            data={"key": "value"},
            observation="Operation completed"
        )
        
        assert result.status == "success"
        assert result.data == {"key": "value"}
        assert str(result) == "Operation completed"
    
    def test_error_result(self):
        """Should create error result."""
        result = ToolResult(
            status="error",
            data={"error": "Something went wrong"},
            observation="Error: Something went wrong"
        )
        
        assert result.status == "error"
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ToolResult(status="success", data=42, observation="Done")
        d = result.to_dict()
        
        assert d["status"] == "success"
        assert d["data"] == 42


class TestWrapToolOutput:
    """Test the wrap_tool_output decorator."""
    
    def test_wrap_success(self):
        """Should wrap successful return in ToolResult."""
        @wrap_tool_output
        def successful_tool():
            return "success!"
        
        result = successful_tool()
        
        assert isinstance(result, ToolResult)
        assert result.status == "success"
        assert result.data == "success!"
    
    def test_wrap_exception(self):
        """Should catch exception and return error ToolResult."""
        @wrap_tool_output
        def failing_tool():
            raise ValueError("Something broke")
        
        result = failing_tool()
        
        assert isinstance(result, ToolResult)
        assert result.status == "error"
        assert "ValueError" in result.observation
    
    def test_wrap_preserves_schema(self):
        """Should preserve __tool_schema__ attribute."""
        @tool_schema
        def schema_tool(x: int) -> int:
            """Tool with schema.
            
            Args:
                x: Input
            """
            return x
        
        wrapped = wrap_tool_output(schema_tool)
        
        assert hasattr(wrapped, "__tool_schema__")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
