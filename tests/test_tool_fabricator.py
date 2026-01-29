"""
Tests for Tool Fabricator - Phase 1

Epic 1, Story 1.2: Tests for Just-in-Time Tool Generation
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_fabricator import (
    ToolFabricator, 
    ToolFabricationError,
    EvolutionarySpecification,
    fabricate_tool
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, response: str = None):
        self.response = response or self._default_tool_code()
        self.prompts_received = []
    
    def _default_tool_code(self) -> str:
        return '''
from core.tool_registry import tool_schema


@tool_schema
def calculate_sum(a: int, b: int) -> int:
    """
    Calculates the sum of two integers.
    
    A simple arithmetic tool for L.O.V.E.
    
    Args:
        a: First integer to add
        b: Second integer to add
        
    Returns:
        The sum of a and b
    """
    try:
        return a + b
    except Exception as e:
        return f"Error: {str(e)}"
'''
    
    async def generate(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        return self.response


class TestToolFabricator:
    """Test the ToolFabricator class."""
    
    @pytest.fixture
    def fabricator(self, tmp_path):
        """Create a fabricator with temp directories."""
        fab = ToolFabricator(llm_client=MockLLMClient())
        fab.incubation_dir = str(tmp_path / "incubation")
        fab.active_dir = str(tmp_path / "active")
        os.makedirs(fab.incubation_dir, exist_ok=True)
        os.makedirs(fab.active_dir, exist_ok=True)
        return fab
    
    @pytest.mark.asyncio
    async def test_fabricate_simple_tool(self, fabricator):
        """Fabrication should create valid Python code."""
        result = await fabricator.fabricate_tool(
            capability_description="Calculate the sum of two numbers",
            tool_name="test_sum_tool"
        )
        
        assert result["success"] is True
        assert result["tool_name"] == "test_sum_tool"
        assert result["location"] == "incubation"
        assert os.path.exists(result["file_path"])
        
        # Verify the file has valid Python
        with open(result["file_path"]) as f:
            code = f.read()
        
        compile(code, "<test>", "exec")
    
    @pytest.mark.asyncio
    async def test_fabricate_to_incubation(self, fabricator):
        """Tools should be written to incubation directory."""
        result = await fabricator.fabricate_tool(
            capability_description="Do something useful"
        )
        
        assert result["success"] is True
        assert "incubation" in result["file_path"]
        assert result["location"] == "incubation"
    
    @pytest.mark.asyncio
    async def test_fabricate_with_specification(self, fabricator):
        """Fabrication from spec should work correctly."""
        spec = EvolutionarySpecification(
            functional_name="multiply_numbers",
            required_arguments={"x": "int", "y": "int"},
            expected_output="int",
            safety_constraints=["pure function"],
            trigger_context="L.O.V.E. needed multiplication"
        )
        
        result = await fabricator.fabricate_from_specification(spec)
        
        assert result["success"] is True
        assert "multiply_numbers" in result["file_path"]
    
    @pytest.mark.asyncio
    async def test_fabricate_invalid_code(self, fabricator):
        """Invalid code should fail validation."""
        # Set up mock to return invalid code
        fabricator.llm_client = MockLLMClient(response="this is not valid python {{{")
        
        result = await fabricator.fabricate_tool(
            capability_description="Something",
            max_retries=0
        )
        
        assert result["success"] is False
        assert "error" in result["message"].lower() or "validation" in result["message"].lower()
    
    def test_promote_tool(self, fabricator):
        """Promoted tool should move from incubation to active."""
        # Create a tool in incubation
        tool_code = '''"""
Auto-fabricated tool: test_promote
Status: INCUBATION - Pending validation and L.O.V.E. approval
"""

from core.tool_registry import tool_schema

@tool_schema
def test_promote(x: int) -> int:
    """Test tool."""
    return x * 2
'''
        incubation_path = os.path.join(fabricator.incubation_dir, "test_promote.py")
        with open(incubation_path, "w") as f:
            f.write(tool_code)
        
        # Promote
        result = fabricator.promote_tool("test_promote")
        
        assert result is True
        assert not os.path.exists(incubation_path)
        assert os.path.exists(os.path.join(fabricator.active_dir, "test_promote.py"))
        
        # Verify status was updated
        with open(os.path.join(fabricator.active_dir, "test_promote.py")) as f:
            code = f.read()
        assert "ACTIVE" in code
    
    def test_list_tools(self, fabricator):
        """Should list tools in incubation and active directories."""
        # Create some test files
        for name in ["tool_a", "tool_b"]:
            with open(os.path.join(fabricator.incubation_dir, f"{name}.py"), "w") as f:
                f.write("# test")
        
        for name in ["tool_c"]:
            with open(os.path.join(fabricator.active_dir, f"{name}.py"), "w") as f:
                f.write("# test")
        
        inc_tools = fabricator.list_incubation_tools()
        active_tools = fabricator.list_active_tools()
        
        assert set(inc_tools) == {"tool_a", "tool_b"}
        assert active_tools == ["tool_c"]
    
    def test_delete_tool(self, fabricator):
        """Should delete tools from specified location."""
        # Create a tool
        path = os.path.join(fabricator.incubation_dir, "to_delete.py")
        with open(path, "w") as f:
            f.write("# test")
        
        result = fabricator.delete_tool("to_delete", location="incubation")
        
        assert result is True
        assert not os.path.exists(path)


class TestEvolutionarySpecification:
    """Test the EvolutionarySpecification dataclass."""
    
    def test_create_spec(self):
        """Should create spec with required fields."""
        spec = EvolutionarySpecification(
            functional_name="test_tool",
            required_arguments={"param1": "str"},
            expected_output="str"
        )
        
        assert spec.functional_name == "test_tool"
        assert spec.status == "pending"
        assert spec.priority == 3
    
    def test_spec_to_dict(self):
        """Should convert to dictionary."""
        spec = EvolutionarySpecification(
            functional_name="test_tool",
            required_arguments={"param1": "str"},
            expected_output="str"
        )
        
        d = spec.to_dict()
        
        assert isinstance(d, dict)
        assert d["functional_name"] == "test_tool"
        assert "created_at" in d


class TestCodeValidation:
    """Test code validation logic."""
    
    @pytest.fixture
    def fabricator(self):
        return ToolFabricator(llm_client=MockLLMClient())
    
    def test_validate_empty_code(self, fabricator):
        """Empty code should fail."""
        result = fabricator._validate_code("")
        assert result["valid"] is False
    
    def test_validate_missing_decorator(self, fabricator):
        """Code without @tool_schema should fail."""
        code = '''
def my_func():
    """Docstring."""
    pass
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is False
        assert "decorator" in result["error"].lower()
    
    def test_validate_missing_docstring(self, fabricator):
        """Code without docstring should fail."""
        code = '''
from core.tool_registry import tool_schema

@tool_schema
def my_func():
    pass
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is False
        assert "docstring" in result["error"].lower()
    
    def test_validate_syntax_error(self, fabricator):
        """Syntax errors should be caught."""
        code = '''
from core.tool_registry import tool_schema

@tool_schema
def my_func(:  # Invalid syntax
    """Docstring."""
    pass
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is False
        assert "syntax" in result["error"].lower()
    
    def test_validate_valid_code(self, fabricator):
        """Valid code should pass."""
        code = '''
from core.tool_registry import tool_schema

@tool_schema
def my_func(x: int) -> int:
    """Does something useful.
    
    Args:
        x: Input value
        
    Returns:
        The result
    """
    return x * 2
'''
        result = fabricator._validate_code(code)
        assert result["valid"] is True


class TestCodeCleaning:
    """Test code cleaning logic."""
    
    @pytest.fixture
    def fabricator(self):
        return ToolFabricator(llm_client=MockLLMClient())
    
    def test_clean_markdown_fences(self, fabricator):
        """Should remove markdown code fences."""
        code = '''```python
def foo():
    pass
```'''
        cleaned = fabricator._clean_code(code)
        assert "```" not in cleaned
        assert "def foo():" in cleaned
    
    def test_clean_plain_fences(self, fabricator):
        """Should remove plain markdown fences."""
        code = '''```
def foo():
    pass
```'''
        cleaned = fabricator._clean_code(code)
        assert "```" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
