
import pytest
import os
from unittest import mock
from core.scientist.test_generator import TestGenerator

@pytest.fixture
def sample_source(tmp_path):
    content = """
def add(a, b):
    return a + b
"""
    f = tmp_path / "calc.py"
    f.write_text(content, encoding="utf-8")
    return str(f)

def test_generate_test_success(sample_source, tmp_path):
    generator = TestGenerator()
    output_file = tmp_path / "test_calc.py"
    
    # Mock run_llm
    # TestGenerator uses asyncio.run(run_llm(...)). run_llm needs to be an async function (or return a coroutine).
    async def mock_run_llm(*args, **kwargs):
        return """
Here is the test:
```python
import pytest
from calc import add

def test_add():
    assert add(1, 2) == 3
```
"""

    with mock.patch("core.scientist.test_generator.run_llm", side_effect=mock_run_llm) as mock_llm:
        # Also need to mock extract_function_metadata or let it run since sample_source is real file
        # It's better to let ast_parser run to verify integration with it.
        
        success = generator.generate_test(sample_source, "add", str(output_file))
        
        assert success
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "def test_add():" in content
        assert "assert add(1, 2) == 3" in content
        
        # Verify prompt contained source
        # Verify prompt contained source
        args, kwargs = mock_llm.call_args
        prompt = kwargs.get("prompt_text")
        if not prompt and args:
             prompt = args[0]
        assert "def add(a, b):" in prompt

def test_generate_test_no_code_block(sample_source, tmp_path):
    generator = TestGenerator()
    output_file = tmp_path / "test_calc.py"
    
    async def mock_run_llm_fail(*args, **kwargs):
        return "I cannot generate a test for this."
    
    with mock.patch("core.scientist.test_generator.run_llm", side_effect=mock_run_llm_fail) as mock_llm:
        
        success = generator.generate_test(sample_source, "add", str(output_file))
        
        assert not success
        assert not output_file.exists()

def test_generate_test_function_not_found(tmp_path):
    generator = TestGenerator()
    # Create empty file
    f = tmp_path / "empty.py"
    f.touch()
    output_file = tmp_path / "test_oops.py"
    
    # extract_function_metadata raises ValueError if not found
    # TestGenerator catches Exception? Yes.
    
    success = generator.generate_test(str(f), "missing_func", str(output_file))
    assert not success

