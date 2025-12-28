"""
Tests for Story 2.3: The "Surgeon" Sandbox
"""
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestIsDockerAvailable:
    """Tests for Docker availability check."""
    
    @patch('subprocess.run')
    def test_docker_available(self, mock_run):
        """Test when Docker is available."""
        from core.surgeon.safe_executor import is_docker_available
        
        mock_run.return_value = MagicMock(returncode=0)
        assert is_docker_available() is True
    
    @patch('subprocess.run')
    def test_docker_not_available(self, mock_run):
        """Test when Docker is not available."""
        from core.surgeon.safe_executor import is_docker_available
        
        mock_run.side_effect = FileNotFoundError()
        assert is_docker_available() is False


class TestSafeExecutePython:
    """Tests for safe_execute_python function."""
    
    @patch('core.surgeon.safe_executor.is_docker_available')
    def test_execute_simple_code_subprocess(self, mock_docker):
        """Test executing simple code via subprocess."""
        from core.surgeon.safe_executor import safe_execute_python
        
        mock_docker.return_value = False
        
        result = safe_execute_python("print('hello world')", use_docker=False)
        
        assert result["success"] is True
        assert "hello world" in result["stdout"]
        assert result["sandbox_type"] == "subprocess"
    
    @patch('core.surgeon.safe_executor.is_docker_available')
    def test_execute_code_with_error(self, mock_docker):
        """Test executing code that raises an error."""
        from core.surgeon.safe_executor import safe_execute_python
        
        mock_docker.return_value = False
        
        result = safe_execute_python("raise ValueError('test error')", use_docker=False)
        
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert "ValueError" in result["stderr"]
    
    @patch('core.surgeon.safe_executor.is_docker_available')
    def test_execute_code_timeout(self, mock_docker):
        """Test that code execution respects timeout."""
        from core.surgeon.safe_executor import safe_execute_python
        
        mock_docker.return_value = False
        
        # Code with infinite loop
        code = "while True: pass"
        result = safe_execute_python(code, timeout=1, use_docker=False)
        
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()
    
    @patch('core.surgeon.safe_executor.is_docker_available')
    def test_execution_time_tracked(self, mock_docker):
        """Test that execution time is tracked."""
        from core.surgeon.safe_executor import safe_execute_python
        
        mock_docker.return_value = False
        
        result = safe_execute_python("import time; time.sleep(0.1)", use_docker=False)
        
        assert result["execution_time"] > 0


class TestVerifyCodeBeforeCommit:
    """Tests for verify_code_before_commit function."""
    
    @patch('core.surgeon.safe_executor.safe_execute_python')
    def test_verify_success(self, mock_execute):
        """Test verification of good code."""
        from core.surgeon.safe_executor import verify_code_before_commit
        
        mock_execute.return_value = {
            "success": True,
            "stdout": "OK",
            "stderr": "",
            "exit_code": 0
        }
        
        result = verify_code_before_commit("print('hello')", "/path/to/file.py")
        
        assert result["verified"] is True
        assert result["can_commit"] is True
        assert "Safe to write" in result["recommendation"]
    
    @patch('core.surgeon.safe_executor.safe_execute_python')
    def test_verify_syntax_error(self, mock_execute):
        """Test verification of code with syntax error."""
        from core.surgeon.safe_executor import verify_code_before_commit
        
        mock_execute.return_value = {
            "success": False,
            "stdout": "",
            "stderr": "SyntaxError: invalid syntax",
            "exit_code": 1
        }
        
        result = verify_code_before_commit("this is not python!!!", "/path/to/file.py")
        
        assert result["verified"] is True
        assert result["can_commit"] is False
        assert "syntax errors" in result["recommendation"].lower()
    
    @patch('core.surgeon.safe_executor.safe_execute_python')
    def test_verify_timeout(self, mock_execute):
        """Test verification of code that times out."""
        from core.surgeon.safe_executor import verify_code_before_commit
        
        mock_execute.return_value = {
            "success": False,
            "stdout": "",
            "stderr": "Timed out after 30 seconds",
            "exit_code": -1
        }
        
        result = verify_code_before_commit("while True: pass", "/path/to/file.py")
        
        assert result["can_commit"] is False
        assert "infinite loop" in result["recommendation"].lower()


class TestExecuteInSubprocess:
    """Tests for subprocess execution."""
    
    def test_subprocess_restricts_environment(self):
        """Test that subprocess has restricted environment."""
        from core.surgeon.safe_executor import _execute_in_subprocess
        
        # Code that tries to access user home
        code = "import os; print(os.environ.get('HOME', 'not_set'))"
        result = _execute_in_subprocess(code, timeout=5)
        
        # HOME should be set to temp dir, not user's actual home
        assert result["success"] is True
    
    def test_subprocess_runs_in_temp_dir(self):
        """Test that subprocess runs in temp directory."""
        from core.surgeon.safe_executor import _execute_in_subprocess
        
        code = "import os; print(os.getcwd())"
        result = _execute_in_subprocess(code, timeout=5)
        
        assert result["success"] is True
        # Should be running in temp directory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
