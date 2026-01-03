
import pytest
import os
from core.surgeon.grafting import graft_function
from core.surgeon.ast_parser import extract_function_metadata

@pytest.fixture
def target_file(tmp_path):
    content = """
class MyClass:
    def method_one(self):
        return 1

    def method_two(self):
        return 2

def standalone():
    print("original")
"""
    f = tmp_path / "graft_target.py"
    f.write_text(content, encoding="utf-8")
    return str(f)

def test_graft_standalone(target_file):
    new_code = """
def standalone():
    print("updated")
    return True
"""
    graft_function(target_file, "standalone", new_code)
    
    # Verify
    meta = extract_function_metadata(target_file, "standalone")
    assert "print(\"updated\")" in meta["source"]
    assert "return True" in meta["source"]

def test_graft_method(target_file):
    new_code = """
def method_one(self):
    # Updated method
    return 100
"""
    graft_function(target_file, "MyClass.method_one", new_code)
    
    # Verify
    meta = extract_function_metadata(target_file, "MyClass.method_one")
    assert "return 100" in meta["source"]
    # Ensure method_two is untouched
    meta2 = extract_function_metadata(target_file, "MyClass.method_two")
    assert "return 2" in meta2["source"]

def test_graft_not_found(target_file):
    new_code = "def ghost(): pass"
    with pytest.raises(ValueError, match="not found"):
        graft_function(target_file, "ghost", new_code)

def test_graft_syntax_error_in_new_code(target_file):
    new_code = "def broken(:"
    with pytest.raises(SyntaxError):
        graft_function(target_file, "standalone", new_code)

def test_mismatch_name_in_replacement(target_file):
    # If the user provides new code with a different function name, 
    # our logic currently tries to find a matching name or takes the first one.
    # Let's test providing a different name, it should still graft into the *target* slot?
    # Wait, the logic finds the node in new_code. If name matches target leaf, good.
    # If not, it takes the first one.
    # Then it replaces the node in the tree (FunctionDef) with this new node.
    # So the name in the file will become the name in the new code.
    
    new_code = """
def renamed_func():
    return "swapped"
"""
    # Replacing 'standalone' with 'renamed_func'
    graft_function(target_file, "standalone", new_code)
    
    # The file should now contain 'renamed_func' instead of 'standalone'
    with pytest.raises(ValueError):
        extract_function_metadata(target_file, "standalone")
        
    meta = extract_function_metadata(target_file, "renamed_func")
    assert meta["function_name"] == "renamed_func"
    assert "return \"swapped\"" in meta["source"]


# =============================================================================
# Story 1.1: CodeGrafter Tests
# =============================================================================

from core.surgeon.ast_parser import CodeGrafter


@pytest.fixture
def grafter():
    """Returns a CodeGrafter instance."""
    return CodeGrafter()


@pytest.fixture
def class_file(tmp_path):
    """Creates a test file with a class for method insertion tests."""
    content = '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self, value: int = 0):
        self.value = value
    
    def add(self, x: int) -> int:
        """Adds x to the current value."""
        self.value += x
        return self.value


class EmptyClass:
    pass
'''
    f = tmp_path / "calculator.py"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def function_file(tmp_path):
    """Creates a test file with functions for body replacement tests."""
    content = '''
def simple_func(a, b):
    """Adds two numbers."""
    return a + b


def multiline_func(items: list) -> dict:
    """Processes items with multiple statements."""
    result = {}
    for i, item in enumerate(items):
        result[i] = item.upper()
    return result


class Processor:
    @staticmethod
    def process(data: str) -> str:
        """Processes data."""
        return data.strip().lower()
    
    @property
    def status(self) -> str:
        """Returns status."""
        return "ready"
'''
    f = tmp_path / "functions.py"
    f.write_text(content, encoding="utf-8")
    return str(f)


# --- insert_class_method Tests ---

def test_insert_simple_method(grafter, class_file):
    """Test inserting a simple method into a class."""
    new_method = '''
def subtract(self, x: int) -> int:
    """Subtracts x from the current value."""
    self.value -= x
    return self.value
'''
    result = grafter.insert_class_method(class_file, "Calculator", new_method)
    
    assert result["success"] is True
    assert result["method_name"] == "subtract"
    
    # Verify the method was inserted
    meta = extract_function_metadata(class_file, "Calculator.subtract")
    assert meta["function_name"] == "subtract"
    assert "self.value -= x" in meta["source"]
    
    # Verify existing methods are untouched
    meta_add = extract_function_metadata(class_file, "Calculator.add")
    assert "self.value += x" in meta_add["source"]


def test_insert_method_into_empty_class(grafter, class_file):
    """Test inserting a method into an empty class."""
    new_method = '''
def first_method(self):
    return "I am the first!"
'''
    result = grafter.insert_class_method(class_file, "EmptyClass", new_method)
    
    assert result["success"] is True
    assert result["method_name"] == "first_method"
    
    # Verify the method was inserted
    meta = extract_function_metadata(class_file, "EmptyClass.first_method")
    assert "I am the first!" in meta["source"]


def test_insert_method_with_decorator(grafter, class_file):
    """Test inserting a method with decorators."""
    new_method = '''
@staticmethod
def multiply(x: int, y: int) -> int:
    """Static method to multiply two numbers."""
    return x * y
'''
    result = grafter.insert_class_method(class_file, "Calculator", new_method)
    
    assert result["success"] is True
    assert result["method_name"] == "multiply"
    
    # Verify the method was inserted with decorator
    meta = extract_function_metadata(class_file, "Calculator.multiply")
    assert "@staticmethod" in meta["source"]
    assert "return x * y" in meta["source"]


def test_insert_method_class_not_found(grafter, class_file):
    """Test error when class is not found."""
    new_method = "def method(self): pass"
    
    with pytest.raises(ValueError, match="not found"):
        grafter.insert_class_method(class_file, "NonExistentClass", new_method)


def test_insert_method_invalid_syntax(grafter, class_file):
    """Test error when method code has invalid syntax."""
    invalid_method = "def broken(:"
    
    with pytest.raises(SyntaxError):
        grafter.insert_class_method(class_file, "Calculator", invalid_method)


def test_insert_method_no_function_def(grafter, class_file):
    """Test error when method code has no function definition."""
    not_a_function = "x = 42"
    
    with pytest.raises(ValueError, match="No function definition"):
        grafter.insert_class_method(class_file, "Calculator", not_a_function)


# --- replace_function_body Tests ---

def test_replace_body_simple(grafter, function_file):
    """Test replacing a simple function body."""
    new_body = '''return a * b  # Now multiplies instead of adds'''
    
    result = grafter.replace_function_body(function_file, "simple_func", new_body)
    
    assert result["success"] is True
    assert "def simple_func(a, b)" in result["preserved_signature"]
    
    # Verify the body was replaced
    meta = extract_function_metadata(function_file, "simple_func")
    assert "return a * b" in meta["source"]
    assert "a + b" not in meta["source"]


def test_replace_body_multiline(grafter, function_file):
    """Test replacing a multi-line function body."""
    new_body = '''
# New implementation
result = {}
for idx, item in enumerate(items):
    # Process differently
    result[f"key_{idx}"] = item.lower()
    result[f"len_{idx}"] = len(item)
return result
'''
    result = grafter.replace_function_body(function_file, "multiline_func", new_body)
    
    assert result["success"] is True
    
    # Verify the body was replaced
    meta = extract_function_metadata(function_file, "multiline_func")
    assert "key_" in meta["source"]
    assert "len_" in meta["source"]
    assert ".upper()" not in meta["source"]  # Old implementation gone


def test_replace_body_preserves_signature(grafter, function_file):
    """Test that replace_function_body preserves the original signature."""
    new_body = '''return "replaced"'''
    
    result = grafter.replace_function_body(function_file, "multiline_func", new_body)
    
    assert result["success"] is True
    # Check that the return type annotation was preserved
    assert "-> dict" in result["preserved_signature"] or "list" in result["preserved_signature"]
    
    # The function should still have its original name
    meta = extract_function_metadata(function_file, "multiline_func")
    assert meta["function_name"] == "multiline_func"


def test_replace_body_class_method(grafter, function_file):
    """Test replacing the body of a class method."""
    new_body = '''return data.upper().strip()  # Changed order'''
    
    result = grafter.replace_function_body(function_file, "Processor.process", new_body)
    
    assert result["success"] is True
    
    # Verify the body was replaced
    meta = extract_function_metadata(function_file, "Processor.process")
    assert ".upper().strip()" in meta["source"]
    assert "@staticmethod" in meta["source"]  # Decorator preserved


def test_replace_body_function_not_found(grafter, function_file):
    """Test error when function is not found."""
    with pytest.raises(ValueError, match="not found"):
        grafter.replace_function_body(function_file, "ghost_function", "pass")


def test_replace_body_invalid_syntax(grafter, function_file):
    """Test error when new body has invalid syntax."""
    invalid_body = "return (("  # Unmatched parenthesis
    
    with pytest.raises(SyntaxError):
        grafter.replace_function_body(function_file, "simple_func", invalid_body)


def test_replace_body_complex_multiline_with_docstring(grafter, tmp_path):
    """Test replacing with a complex multi-line body including docstrings and comments."""
    # Create a file with a function to modify
    content = '''
def complex_func(x: int, y: int, z: int = 0) -> dict:
    """Original docstring."""
    return {"sum": x + y + z}
'''
    f = tmp_path / "complex.py"
    f.write_text(content, encoding="utf-8")
    target = str(f)
    
    new_body = '''
"""New docstring that replaces the original.

This is a multi-line docstring with details.
"""
# Step 1: Validate inputs
if x < 0 or y < 0:
    raise ValueError("Inputs must be non-negative")

# Step 2: Compute result
result = {
    "product": x * y,
    "adjusted": (x * y) + z,
    "metadata": {
        "inputs": [x, y, z]
    }
}

# Step 3: Return
return result
'''
    result = grafter.replace_function_body(target, "complex_func", new_body)
    
    assert result["success"] is True
    
    # Verify the complex body was inserted correctly
    meta = extract_function_metadata(target, "complex_func")
    assert "New docstring" in meta["source"]
    assert "Step 1: Validate" in meta["source"]
    assert "product" in meta["source"]
    assert "metadata" in meta["source"]
    
    # Original signature preserved
    assert meta["function_name"] == "complex_func"
    assert meta["return_type"] == "dict"

