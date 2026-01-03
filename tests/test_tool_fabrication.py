"""
Tests for Epic 4: The Principle of Morphic Complexity (Tool Fabrication)

Story 4.1: Just-in-Time Tool Fabrication
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock

from core.tool_registry import ToolRegistry, tool_schema, get_global_registry
from core.tool_fabricator import ToolFabricator, ToolFabricationError, fabricate_tool


# =============================================================================
# Story 4.1: Dynamic Tool Loading Tests
# =============================================================================

class TestToolRegistryDynamicLoading:
    """Tests for the dynamic tool loading functionality in ToolRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh ToolRegistry for each test."""
        return ToolRegistry()
    
    @pytest.fixture
    def temp_custom_dir(self, tmp_path):
        """Create a temporary custom tools directory."""
        custom_dir = tmp_path / "custom_tools"
        custom_dir.mkdir()
        return custom_dir
    
    def test_load_custom_tools_empty_directory(self, registry, temp_custom_dir):
        """Loading from empty directory should return empty list."""
        result = registry.load_custom_tools(str(temp_custom_dir))
        assert result == []
    
    def test_load_custom_tools_with_valid_tool(self, registry, temp_custom_dir):
        """Valid tool files should be loaded and registered."""
        # Create a valid tool file
        tool_code = '''
from core.tool_registry import tool_schema

@tool_schema
def sample_custom_tool(input_text: str) -> str:
    """
    A sample custom tool for testing.
    
    Args:
        input_text: Text to process
        
    Returns:
        Processed text
    """
    return f"Processed: {input_text}"
'''
        tool_file = temp_custom_dir / "sample_custom_tool.py"
        tool_file.write_text(tool_code)
        
        result = registry.load_custom_tools(str(temp_custom_dir))
        
        assert "sample_custom_tool" in result
        assert "sample_custom_tool" in registry
    
    def test_load_custom_tools_skips_init_files(self, registry, temp_custom_dir):
        """Files starting with _ should be skipped."""
        init_file = temp_custom_dir / "__init__.py"
        init_file.write_text("# Init file")
        
        result = registry.load_custom_tools(str(temp_custom_dir))
        
        assert result == []
    
    def test_refresh_clears_and_reloads(self, registry, temp_custom_dir):
        """Refresh should clear custom tools and reload them."""
        # Create initial tool
        tool_code = '''
from core.tool_registry import tool_schema

@tool_schema
def refreshable_tool(x: int) -> str:
    """Test tool.
    
    Args:
        x: A number
        
    Returns:
        String result
    """
    return str(x)
'''
        tool_file = temp_custom_dir / "refreshable_tool.py"
        tool_file.write_text(tool_code)
        
        # Patch the CUSTOM_TOOLS_DIR
        registry.CUSTOM_TOOLS_DIR = str(temp_custom_dir)
        
        # Load once
        registry.load_custom_tools(str(temp_custom_dir))
        assert "refreshable_tool" in registry
        
        # Refresh
        result = registry.refresh()
        
        assert "loaded" in result
        assert "total" in result
    
    def test_unregister_removes_tool(self, registry):
        """Unregister should remove a tool from the registry."""
        @tool_schema
        def removable_tool(x: int) -> str:
            """A tool to remove.
            
            Args:
                x: A number
                
            Returns:
                String result
            """
            return str(x)
        
        registry.register(removable_tool)
        assert "removable_tool" in registry
        
        result = registry.unregister("removable_tool")
        
        assert result is True
        assert "removable_tool" not in registry
    
    def test_unregister_nonexistent_returns_false(self, registry):
        """Unregistering a nonexistent tool should return False."""
        result = registry.unregister("nonexistent_tool")
        assert result is False
    
    def test_get_custom_tools(self, registry, temp_custom_dir):
        """get_custom_tools should return only dynamically loaded tools."""
        # Register a native tool
        @tool_schema
        def native_tool(x: int) -> str:
            """Native tool.
            
            Args:
                x: A number
                
            Returns:
                String result
            """
            return str(x)
        
        registry.register(native_tool)
        
        # Load a custom tool
        tool_code = '''
from core.tool_registry import tool_schema

@tool_schema
def dynamic_tool(s: str) -> str:
    """Dynamic tool.
    
    Args:
        s: A string
        
    Returns:
        String result
    """
    return s.upper()
'''
        tool_file = temp_custom_dir / "dynamic_tool.py"
        tool_file.write_text(tool_code)
        
        registry.load_custom_tools(str(temp_custom_dir))
        
        custom_tools = registry.get_custom_tools()
        
        assert "dynamic_tool" in custom_tools
        assert "native_tool" not in custom_tools


# =============================================================================
# Story 4.1: Tool Fabricator Tests
# =============================================================================

class TestToolFabricator:
    """Tests for the ToolFabricator class."""
    
    @pytest.fixture
    def fabricator(self, tmp_path):
        """Create a ToolFabricator with temp directory."""
        custom_dir = tmp_path / "tools" / "custom"
        custom_dir.mkdir(parents=True)
        
        fab = ToolFabricator()
        fab.custom_dir = str(custom_dir)
        return fab
    
    def test_extract_tool_name_with_decorator(self, fabricator):
        """Should extract tool name from @tool_schema decorated function."""
        code = '''
@tool_schema
def my_awesome_tool(x: int) -> str:
    """Doc."""
    return str(x)
'''
        name = fabricator._extract_tool_name(code)
        assert name == "my_awesome_tool"
    
    def test_extract_tool_name_plain_function(self, fabricator):
        """Should extract tool name from plain function definition."""
        code = '''
def simple_function(x: int) -> str:
    return str(x)
'''
        name = fabricator._extract_tool_name(code)
        assert name == "simple_function"
    
    def test_sanitize_tool_name(self, fabricator):
        """Should convert description to valid Python identifier."""
        assert fabricator._sanitize_tool_name("Parse PDF Files") == "parse_pdf_files"
        assert fabricator._sanitize_tool_name("123 Numbers") == "tool_123_numbers"
        assert fabricator._sanitize_tool_name("Special!@#$Chars") == "specialchars"
    
    def test_validate_code_valid(self, fabricator):
        """Should validate syntactically correct code."""
        code = '''
from core.tool_registry import tool_schema

@tool_schema
def valid_tool(x: int) -> str:
    """Doc."""
    return str(x)
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is True
        assert result["error"] is None
    
    def test_validate_code_syntax_error(self, fabricator):
        """Should catch syntax errors."""
        code = '''
@tool_schema
def broken(
    # Missing closing paren
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is False
        assert result["error"] is not None
    
    def test_validate_code_missing_decorator(self, fabricator):
        """Should require @tool_schema decorator."""
        code = '''
def missing_decorator(x: int) -> str:
    return str(x)
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is False
        assert "Missing @tool_schema" in result["error"]
    
    def test_validate_code_empty(self, fabricator):
        """Should reject empty code."""
        result = fabricator._validate_code("")
        assert result["valid"] is False
        assert "Empty code" in result["error"]
    
    @pytest.mark.asyncio
    async def test_fabricate_tool_success(self, fabricator):
        """Should successfully fabricate and save a tool."""
        # Mock the LLM to return valid tool code
        mock_code = '''
from core.tool_registry import tool_schema

@tool_schema
def string_reverser(text: str) -> str:
    """
    Reverses a string.
    
    Args:
        text: The text to reverse
        
    Returns:
        The reversed text
    """
    try:
        return text[::-1]
    except Exception as e:
        return f"Error: {str(e)}"
'''
        
        async def mock_llm(prompt):
            return {"result": mock_code}
        
        fabricator.llm_runner = mock_llm
        
        # Mock the registry refresh
        with patch('core.tool_registry.get_global_registry') as mock_registry:
            mock_reg_instance = MagicMock()
            mock_reg_instance.refresh.return_value = {"loaded": ["string_reverser"]}
            mock_reg_instance.__contains__ = lambda s, x: x == "string_reverser"
            mock_registry.return_value = mock_reg_instance
            
            result = await fabricator.fabricate_tool("Reverse a string")
        
        assert result["success"] is True
        assert result["tool_name"] == "string_reverser"
        assert os.path.exists(result["file_path"])
    
    @pytest.mark.asyncio
    async def test_fabricate_tool_handles_markdown_fences(self, fabricator):
        """Should handle LLM output with markdown fences."""
        mock_code = '''```python
from core.tool_registry import tool_schema

@tool_schema
def fenced_tool(x: int) -> str:
    """
    A tool wrapped in markdown.
    
    Args:
        x: A number
        
    Returns:
        String result
    """
    return str(x)
```'''
        
        async def mock_llm(prompt):
            return {"result": mock_code}
        
        fabricator.llm_runner = mock_llm
        
        with patch('core.tool_registry.get_global_registry') as mock_registry:
            mock_reg_instance = MagicMock()
            mock_reg_instance.refresh.return_value = {"loaded": ["fenced_tool"]}
            mock_reg_instance.__contains__ = lambda s, x: x == "fenced_tool"
            mock_registry.return_value = mock_reg_instance
            
            result = await fabricator.fabricate_tool("A simple tool")
        
        assert result["success"] is True
    
    def test_list_fabricated_tools(self, fabricator):
        """Should list all tools in custom directory."""
        # Create some tool files
        (fabricator.custom_dir + "/tool_a.py").replace("/", os.sep)
        tool_a = os.path.join(fabricator.custom_dir, "tool_a.py")
        tool_b = os.path.join(fabricator.custom_dir, "tool_b.py")
        
        with open(tool_a, "w") as f:
            f.write("# Tool A")
        with open(tool_b, "w") as f:
            f.write("# Tool B")
        
        tools = fabricator.list_fabricated_tools()
        
        assert "tool_a" in tools
        assert "tool_b" in tools
    
    def test_delete_tool(self, fabricator):
        """Should delete a fabricated tool."""
        tool_file = os.path.join(fabricator.custom_dir, "deletable.py")
        
        with open(tool_file, "w") as f:
            f.write("# Deletable tool")
        
        assert os.path.exists(tool_file)
        
        with patch('core.tool_registry.get_global_registry') as mock_registry:
            mock_reg_instance = MagicMock()
            mock_reg_instance.unregister.return_value = True
            mock_registry.return_value = mock_reg_instance
            
            result = fabricator.delete_tool("deletable")
        
        assert result is True
        assert not os.path.exists(tool_file)


# =============================================================================
# Integration Tests
# =============================================================================

class TestToolFabricationIntegration:
    """Integration tests for the tool fabrication workflow."""
    
    def test_global_registry_has_custom_tools_dir(self):
        """Global registry should have CUSTOM_TOOLS_DIR defined."""
        registry = get_global_registry()
        assert hasattr(registry, "CUSTOM_TOOLS_DIR")
    
    def test_global_registry_has_refresh_method(self):
        """Global registry should have refresh method."""
        registry = get_global_registry()
        assert hasattr(registry, "refresh")
        assert callable(registry.refresh)
    
    def test_global_registry_has_load_custom_tools_method(self):
        """Global registry should have load_custom_tools method."""
        registry = get_global_registry()
        assert hasattr(registry, "load_custom_tools")
        assert callable(registry.load_custom_tools)
