
import pytest
import os
import tempfile
from core.surgeon.safe_executor import (
    preflight_check,
    check_syntax,
    ForbiddenMutationError,
    _suggest_syntax_remediation,
    _suggest_smoke_test_remediation
)


# =============================================================================
# Story 1.2: Preflight Check Tests
# =============================================================================

class TestPreflightCheck:
    """Tests for the preflight_check function."""
    
    def test_preflight_valid_code(self):
        """Test that valid code passes preflight check."""
        valid_code = '''
def hello():
    """Says hello."""
    return "Hello, World!"

x = hello()
print(x)
'''
        result = preflight_check(valid_code, run_smoke_tests=False)
        
        assert result["passed"] is True
        assert result["compile_result"]["success"] is True
        assert result["structured_error"] is None
    
    def test_preflight_syntax_error(self):
        """Test that syntax errors are caught and structured correctly."""
        invalid_code = '''
def broken_func(
    return "oops"
'''
        result = preflight_check(invalid_code, run_smoke_tests=False)
        
        assert result["passed"] is False
        assert result["compile_result"]["success"] is False
        assert result["compile_result"]["error"] is not None
        assert result["compile_result"]["line"] is not None
        
        # Check structured error for Reasoning node
        assert result["structured_error"] is not None
        assert result["structured_error"]["error_type"] == "SyntaxError"
        assert "message" in result["structured_error"]
        assert "remediation" in result["structured_error"]
        assert "context" in result["structured_error"]
    
    def test_preflight_structured_error_format(self):
        """Verify the exact format of structured errors for Reasoning node."""
        invalid_code = "def test(:\n    pass"
        
        result = preflight_check(invalid_code, run_smoke_tests=False)
        
        error = result["structured_error"]
        assert error is not None
        
        # Check all required fields
        assert "error_type" in error
        assert "message" in error
        assert "remediation" in error
        assert "context" in error
        
        # Check context structure
        context = error["context"]
        assert "line" in context
        assert "code_snippet" in context or "column" in context
    
    def test_preflight_with_target_file(self):
        """Test preflight check with a target file path."""
        valid_code = "x = 42"
        
        result = preflight_check(
            valid_code, 
            target_file="/path/to/test.py",
            run_smoke_tests=False
        )
        
        assert result["passed"] is True
        assert result["compile_result"]["success"] is True
    
    def test_preflight_multiline_syntax_error(self):
        """Test handling of multi-line syntax errors."""
        # Missing closing bracket
        code = '''
data = [
    1,
    2,
    3
# Missing closing bracket
'''
        result = preflight_check(code, run_smoke_tests=False)
        
        assert result["passed"] is False
        assert result["structured_error"]["error_type"] == "SyntaxError"
    
    def test_preflight_indent_error(self):
        """Test handling of indentation errors."""
        code = '''
def test():
return "bad indent"
'''
        result = preflight_check(code, run_smoke_tests=False)
        
        assert result["passed"] is False
        assert result["compile_result"]["success"] is False


class TestSyntaxRemediation:
    """Tests for syntax error remediation suggestions."""
    
    def test_eof_suggestion(self):
        """Test remediation for unexpected EOF."""
        class MockError:
            msg = "unexpected EOF while parsing"
            lineno = 1
            offset = 1
            text = "("
        
        suggestion = _suggest_syntax_remediation(MockError())
        assert "closing" in suggestion.lower() or "bracket" in suggestion.lower()
    
    def test_indent_suggestion(self):
        """Test remediation for indentation errors."""
        class MockError:
            msg = "unexpected indent"
            lineno = 2
            offset = 4
            text = "    pass"
        
        suggestion = _suggest_syntax_remediation(MockError())
        assert "indent" in suggestion.lower()
    
    def test_generic_suggestion(self):
        """Test generic remediation when no specific pattern matches."""
        class MockError:
            msg = "some weird error"
            lineno = 1
            offset = 1
            text = "???"
        
        suggestion = _suggest_syntax_remediation(MockError())
        assert len(suggestion) > 0  # Should provide some suggestion


class TestSmokeTestRemediation:
    """Tests for smoke test failure remediation suggestions."""
    
    def test_import_error_suggestion(self):
        """Test remediation for import errors."""
        result = {"stderr": "ImportError: No module named 'nonexistent'", "exit_code": 1}
        suggestion = _suggest_smoke_test_remediation(result)
        assert "module" in suggestion.lower() or "import" in suggestion.lower()
    
    def test_attribute_error_suggestion(self):
        """Test remediation for attribute errors."""
        result = {"stderr": "AttributeError: 'str' object has no attribute 'foo'", "exit_code": 1}
        suggestion = _suggest_smoke_test_remediation(result)
        assert "attribute" in suggestion.lower()
    
    def test_type_error_suggestion(self):
        """Test remediation for type errors."""
        result = {"stderr": "TypeError: expected str, got int", "exit_code": 1}
        suggestion = _suggest_smoke_test_remediation(result)
        assert "type" in suggestion.lower() or "argument" in suggestion.lower()
    
    def test_timeout_suggestion(self):
        """Test remediation for timeout errors."""
        result = {"stderr": "Smoke test timed out", "exit_code": -1}
        suggestion = _suggest_smoke_test_remediation(result)
        assert "loop" in suggestion.lower() or "performance" in suggestion.lower()


class TestCheckSyntax:
    """Tests for the check_syntax function."""
    
    def test_valid_syntax(self):
        """Test that valid code passes syntax check."""
        result = check_syntax("x = 42")
        assert result["valid"] is True
        assert result["error"] is None
    
    def test_invalid_syntax(self):
        """Test that invalid code fails syntax check."""
        result = check_syntax("def broken(:")
        assert result["valid"] is False
        assert result["error"] is not None
        assert result["line"] is not None
    
    def test_complex_valid_code(self):
        """Test syntax check with complex valid code."""
        code = '''
class MyClass:
    def __init__(self, value: int = 0):
        self.value = value
    
    @property
    def doubled(self) -> int:
        return self.value * 2

async def async_func():
    await something()
    return True
'''
        result = check_syntax(code)
        assert result["valid"] is True


class TestForbiddenMutationError:
    """Tests for the ForbiddenMutationError exception."""
    
    def test_exception_can_be_raised(self):
        """Test that the exception can be raised and caught."""
        with pytest.raises(ForbiddenMutationError):
            raise ForbiddenMutationError("Cannot modify core/guardian/safety.py")
    
    def test_exception_message(self):
        """Test that the exception includes the message."""
        try:
            raise ForbiddenMutationError("Test message")
        except ForbiddenMutationError as e:
            assert "Test message" in str(e)
