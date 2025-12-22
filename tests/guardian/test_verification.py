
import pytest
from unittest import mock
from core.guardian.verification import VerificationPipeline

class MockSandbox:
    def __init__(self):
        self.last_command = None
        # Default behavior: Success on everything
        self.return_code = 0
        self.stdout = ""
        self.stderr = ""

    def run_command(self, command):
        self.last_command = command
        # Simulate failure for specific commands if needed by test logic
        if "fail_test" in command:
             return 1, "Tests failed", "Error"
        if "lint_fail" in command:
             return 1, "Lint errors", ""
             
        return self.return_code, self.stdout, self.stderr

@pytest.fixture
def pipeline():
    return VerificationPipeline(sandbox=MockSandbox())

def test_verify_syntax_valid(pipeline):
    code = "def foo(): pass"
    assert pipeline.verify_syntax(code)

def test_verify_syntax_invalid(pipeline):
    code = "def foo(): return @"  # Invalid syntax
    assert not pipeline.verify_syntax(code)

def test_verify_semantics_success(pipeline):
    # Default mock succeeds
    assert pipeline.verify_semantics("tests/test_good.py")
    assert "pytest tests/test_good.py" in pipeline.sandbox.last_command

def test_verify_semantics_failure(pipeline):
    assert not pipeline.verify_semantics("tests/fail_test.py")

def test_verify_style_success(pipeline):
    assert pipeline.verify_style("core/good.py")
    assert "ruff check core/good.py" in pipeline.sandbox.last_command

def test_verify_style_failure(pipeline):
    assert not pipeline.verify_style("core/lint_fail.py")

def test_verify_all_flow(pipeline, tmp_path):
    # We need a real file for syntax check part of verify_all
    f = tmp_path / "valid.py"
    f.write_text("print('hello')", encoding="utf-8")
    
    # Mock relative path usage by passing absolute path to verify_all since our mock sandbox accepts anything
    # but the 'os.path.exists' check in verify_all needs to find it.
    
    assert pipeline.verify_all(str(f), "tests/fake_test.py")

