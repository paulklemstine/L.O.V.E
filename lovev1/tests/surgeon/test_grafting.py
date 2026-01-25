
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
