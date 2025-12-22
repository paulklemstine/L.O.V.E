
import pytest
import os
from unittest import mock
from core.scientist.mutation import MutationEngine

@pytest.fixture
def sample_file(tmp_path):
    content = """
def calculate(a: int, b: int) -> int:
    return a + b
"""
    f = tmp_path / "math_lib.py"
    f.write_text(content, encoding="utf-8")
    return str(f)

def test_evolve_function_success(sample_file):
    engine = MutationEngine()
    
    # Correct signature, different body
    new_code = """
def calculate(a: int, b: int) -> int:
    # Optimized
    return a + b
"""
    
    # Mock LLM response
    # Async mock wrapper needed
    async def mock_run_llm(*args, **kwargs):
        return f"```python\n{new_code}\n```"
        
    with mock.patch("core.scientist.mutation.run_llm", side_effect=mock_run_llm):
        result = engine.evolve_function(sample_file, "calculate", "Add comment")
        
        assert result is not None
        assert "# Optimized" in result
        assert "def calculate" in result

def test_evolve_function_signature_mismatch(sample_file):
    engine = MutationEngine()
    
    # WRONG signature (variable name changed 'b' to 'c')
    bad_code = """
def calculate(a: int, c: int) -> int:
    return a + c
"""
    
    async def mock_run_llm(*args, **kwargs):
        return f"```python\n{bad_code}\n```"
        
    with mock.patch("core.scientist.mutation.run_llm", side_effect=mock_run_llm):
        result = engine.evolve_function(sample_file, "calculate", "Rename var")
        
        # Should be rejected
        assert result is None

def test_evolve_function_type_annotation_mismatch(sample_file):
    engine = MutationEngine()
    
    # WRONG return type
    bad_code = """
def calculate(a: int, b: int) -> float:
    return float(a + b)
"""
    
    async def mock_run_llm(*args, **kwargs):
        return f"```python\n{bad_code}\n```"
        
    with mock.patch("core.scientist.mutation.run_llm", side_effect=mock_run_llm):
        result = engine.evolve_function(sample_file, "calculate", "Change return type")
        
        assert result is None
