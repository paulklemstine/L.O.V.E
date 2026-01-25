
import pytest
from unittest.mock import MagicMock, patch, mock_open
from core.guardian.verification import VerificationPipeline

@pytest.fixture
def pipeline():
    # We patch the factory 'get_sandbox' to return a mock
    with patch('core.guardian.verification.get_sandbox') as mock_get_sandbox:
        mock_sandbox_instance = MagicMock()
        mock_get_sandbox.return_value = mock_sandbox_instance
        yield VerificationPipeline()

def test_verify_syntax_valid(pipeline):
    code = "def foo(): pass"
    assert pipeline.verify_syntax(code)

def test_verify_syntax_invalid(pipeline):
    code = "def foo(): return @"  # Invalid syntax
    assert not pipeline.verify_syntax(code)

def test_verify_semantics_success(pipeline):
    pipeline.sandbox.run_command.return_value = (0, "Success", "")
    assert pipeline.verify_semantics("tests/my_test.py") is True
    pipeline.sandbox.run_command.assert_called_with("python3 -m pytest tests/my_test.py")

def test_verify_semantics_failure(pipeline):
    pipeline.sandbox.run_command.return_value = (1, "", "Test failed")
    assert pipeline.verify_semantics("tests/my_test.py") is False

def test_verify_style_success(pipeline):
    pipeline.sandbox.run_command.return_value = (0, "", "")
    assert pipeline.verify_style("my_file.py") is True
    pipeline.sandbox.run_command.assert_called_with("ruff check my_file.py")

def test_verify_style_failure(pipeline):
    pipeline.sandbox.run_command.return_value = (1, "Lint error", "")
    assert pipeline.verify_style("my_file.py") is False

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data="def f(): pass")
def test_verify_all_success(mock_file, mock_exists, pipeline):
    pipeline.sandbox.run_command.return_value = (0, "Success", "")
    assert pipeline.verify_all("target.py", "test.py") is True

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data="def f(): pass")
def test_verify_all_semantic_fail(mock_file, mock_exists, pipeline):
    pipeline.sandbox.run_command.side_effect = [
        (1, "", "Test failed"),
        (0, "", "")
    ]
    assert pipeline.verify_all("target.py", "test.py") is False

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data="def f() no")
def test_verify_all_syntax_fail(mock_file, mock_exists, pipeline):
    assert pipeline.verify_all("target.py", "test.py") is False

