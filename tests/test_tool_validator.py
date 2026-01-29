"""
Tests for Tool Validator - Phase 2

Epic 1, Story 1.3: Tests for Automated Qualification Pipeline
"""

import os
import sys
import pytest
import asyncio
from typing import Tuple 

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_validator import ToolValidator, ValidationResult


class MockSandbox:
    """Mock sandbox for testing."""
    def __init__(self, exit_code=0, stdout="1 passed", stderr=""):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.commands = []
        
    def run_command(self, command, **kwargs):
        self.commands.append(command)
        return self.exit_code, self.stdout, self.stderr


class MockLLMClient:
    """Mock LLM client for testing."""
    async def generate(self, prompt: str) -> str:
        return """
import pytest

def test_tool():
    assert True
"""


class TestToolValidator:
    """Test the ToolValidator class."""
    
    @pytest.fixture
    def validator(self):
        return ToolValidator(
            sandbox=MockSandbox(),
            llm_client=MockLLMClient()
        )
    
    def test_syntax_check_valid(self, validator):
        """Should pass valid python syntax."""
        code = "def foo(): pass"
        assert validator._check_syntax(code) is True
    
    def test_syntax_check_invalid(self, validator):
        """Should fail invalid python syntax."""
        code = "def foo(): : pass"
        assert validator._check_syntax(code) is False
        
    def test_security_scan_clean(self, validator):
        """Should pass clean code."""
        code = "import json\ndef foo(): return json.dumps({})"
        issues = validator._security_scan(code)
        assert len(issues) == 0
        
    def test_security_scan_dangerous(self, validator):
        """Should flag dangerous imports."""
        code = "import os\ndef kill(): os.system('rm -rf /')"
        issues = validator._security_scan(code)
        assert len(issues) > 0
        assert "os.system" in issues[0]
    
    @pytest.mark.asyncio
    async def test_validate_success(self, validator, tmp_path):
        """Should pass validation when all checks pass."""
        # Create a dummy tool file
        tool_path = tmp_path / "my_tool.py"
        with open(tool_path, "w") as f:
            f.write("def my_tool(): pass")
            
        # Configure sandbox to return success
        validator.sandbox = MockSandbox(stdout="1 passed in 0.1s")
        
        result = await validator.validate(
            tool_name="my_tool",
            tool_path=str(tool_path),
            tool_code="def my_tool(): pass"
        )
        
        assert result.passed is True
        assert result.syntax_valid is True
        assert result.tests_passed == 1
    
    @pytest.mark.asyncio
    async def test_validate_syntax_fail(self, validator, tmp_path):
        """Should fail fast on syntax error."""
        tool_path = tmp_path / "broken_tool.py"
        with open(tool_path, "w") as f:
            f.write("Broken code")
            
        result = await validator.validate(
            tool_name="broken_tool", 
            tool_path=str(tool_path)
        )
        
        assert result.passed is False
        assert result.syntax_valid is False
        assert "Syntax" in result.error_message

    @pytest.mark.asyncio
    async def test_validate_tests_fail(self, validator, tmp_path):
        """Should fail if sandbox tests fail."""
        tool_path = tmp_path / "fail_tool.py"
        with open(tool_path, "w") as f:
            f.write("def fail_tool(): pass")
            
        # Configure sandbox to return failure
        validator.sandbox = MockSandbox(exit_code=1, stdout="1 failed")
        
        result = await validator.validate(
            tool_name="fail_tool",
            tool_path=str(tool_path)
        )
        
        assert result.passed is False
        assert result.tests_passed == 0
        assert "Tests failed" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
