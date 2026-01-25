
import pytest
import os
from core.surgeon.ast_parser import extract_function_metadata

# Helper to create a temp file
@pytest.fixture
def sample_file(tmp_path):
    content = """
import logging

def simple_func(a, b):
    \"\"\"Docstring.\"\"\"
    return a + b

class MyClass:
    def method_one(self, x: int) -> int:
        # A comment
        return x * 2

    @staticmethod
    def static_method():
        pass

def outer():
    def inner():
        return "nested"
    return inner
"""
    f = tmp_path / "test_code.py"
    f.write_text(content, encoding="utf-8")
    return str(f)

@pytest.fixture
def bad_syntax_file(tmp_path):
    content = """
def broken_func(
    return "oops"
"""
    f = tmp_path / "bad_code.py"
    f.write_text(content, encoding="utf-8")
    return str(f)

def test_extract_simple_func(sample_file):
    meta = extract_function_metadata(sample_file, "simple_func")
    assert meta["function_name"] == "simple_func"
    assert "return a + b" in meta["source"]
    assert meta["start_line"] == 4
    assert meta["args"] == ["a", "b"]

def test_extract_class_method(sample_file):
    meta = extract_function_metadata(sample_file, "MyClass.method_one")
    assert meta["function_name"] == "method_one"
    assert "return x * 2" in meta["source"]
    assert meta["return_type"] == "int"
    assert meta["args"] == ["self", "x: int"]

def test_extract_static_method(sample_file):
    meta = extract_function_metadata(sample_file, "MyClass.static_method")
    assert meta["function_name"] == "static_method"
    assert "@staticmethod" in meta["source"]

def test_extract_nested(sample_file):
    # Depending on implementation, "inner" might be reachable as "outer.inner" if we tracked scope stack correctly?
    # Our impl tracks ClassDef in scope, but FunctionDef? 
    # Let's check the code: visit_FunctionDef pushes to scope stack *after* checking match.
    # So "inner" inside "outer".
    # visit_FunctionDef(outer) -> match(outer)? checks target name. push outer.
    # visit_FunctionDef(inner) -> scope is "outer". full_name "outer.inner".
    
    meta = extract_function_metadata(sample_file, "outer.inner")
    assert meta["function_name"] == "inner"
    assert "return \"nested\"" in meta["source"]

def test_not_found(sample_file):
    with pytest.raises(ValueError, match="not found"):
        extract_function_metadata(sample_file, "non_existent")

def test_syntax_error(bad_syntax_file):
    with pytest.raises(SyntaxError):
        extract_function_metadata(bad_syntax_file, "broken_func")

