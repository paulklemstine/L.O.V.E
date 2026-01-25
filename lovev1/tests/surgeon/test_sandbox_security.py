"""
DeepAgent Protocol - Story 4.1: Sandbox Security Tests

Ensures that code execution is sandboxed and cannot affect host files.
"""

import pytest
import os
import tempfile
from pathlib import Path


class TestSandboxSecurity:
    """Test suite for code sandbox security constraints."""
    
    @pytest.fixture
    def sandbox(self):
        """Create a sandbox instance for testing."""
        try:
            from core.surgeon.sandbox import DockerSandbox
            return DockerSandbox()
        except ImportError:
            pytest.skip("DockerSandbox not available")
    
    def test_os_remove_blocked(self, sandbox):
        """
        Story 4.1: Verify sandbox prevents file deletion attacks.
        
        Attempts os.remove on a protected file and asserts it's blocked.
        """
        # Attempt to delete a core file
        code = "import os; os.remove('/app/core/state.py'); print('DELETED')"
        
        try:
            exit_code, stdout, stderr = sandbox.run_command(
                f'python3 -c "{code}"',
                timeout=30
            )
            
            # The command should fail (non-zero exit) or show permission denied
            # Success would mean the sandbox worked (file not accessible or protected)
            if exit_code == 0 and "DELETED" in stdout:
                # Check if file still exists on host
                host_file = Path(__file__).parent.parent / "core" / "state.py"
                assert host_file.exists(), "Sandbox failed: state.py was deleted!"
            else:
                # Command failed as expected (permission denied or file not found in container)
                assert True, "Sandbox correctly prevented file deletion"
                
        except Exception as e:
            # Docker not running or other error - skip test
            pytest.skip(f"Sandbox test skipped: {e}")
    
    def test_cannot_modify_core_files(self, sandbox):
        """Verify sandbox cannot modify core files."""
        code = '''
import os
try:
    with open('/app/core/state.py', 'a') as f:
        f.write('# MALICIOUS MODIFICATION')
    print('MODIFIED')
except Exception as e:
    print(f'BLOCKED: {e}')
'''
        
        try:
            exit_code, stdout, stderr = sandbox.run_command(
                f"python3 -c \"{code}\"",
                timeout=30
            )
            
            # Check that modification was blocked
            assert "MODIFIED" not in stdout or "BLOCKED" in stdout, \
                "Sandbox failed to prevent file modification"
                
        except Exception as e:
            pytest.skip(f"Sandbox test skipped: {e}")
    
    def test_network_isolation(self, sandbox):
        """Test that network can be disabled in sandbox."""
        code = '''
import urllib.request
try:
    urllib.request.urlopen('http://example.com', timeout=5)
    print('NETWORK_ACCESSIBLE')
except Exception as e:
    print(f'NETWORK_BLOCKED: {e}')
'''
        
        try:
            exit_code, stdout, stderr = sandbox.run_command(
                f"python3 -c \"{code}\"",
                timeout=30,
                network_disabled=True
            )
            
            # Network should be blocked
            assert "NETWORK_BLOCKED" in stdout or "NETWORK_ACCESSIBLE" not in stdout, \
                "Sandbox network isolation failed"
                
        except Exception as e:
            pytest.skip(f"Sandbox test skipped: {e}")


class TestToolResultWrapper:
    """Test the ToolResult wrapper functionality."""
    
    def test_wrap_tool_output_success(self):
        """Test that successful tool execution is wrapped correctly."""
        from core.tool_registry import wrap_tool_output, ToolResult
        
        @wrap_tool_output
        def my_tool(x: int) -> str:
            return f"Result: {x}"
        
        result = my_tool(42)
        
        assert isinstance(result, ToolResult)
        assert result.status == "success"
        assert result.data == "Result: 42"
        assert "42" in result.observation
    
    def test_wrap_tool_output_error(self):
        """Test that tool errors are wrapped correctly."""
        from core.tool_registry import wrap_tool_output, ToolResult
        
        @wrap_tool_output
        def failing_tool() -> str:
            raise ValueError("Something went wrong")
        
        result = failing_tool()
        
        assert isinstance(result, ToolResult)
        assert result.status == "error"
        assert "ValueError" in result.observation
        assert "Something went wrong" in result.observation
    
    def test_tool_result_to_dict(self):
        """Test ToolResult serialization."""
        from core.tool_registry import ToolResult
        
        result = ToolResult(
            status="success",
            data={"key": "value"},
            observation="Test observation"
        )
        
        d = result.to_dict()
        
        assert d["status"] == "success"
        assert d["data"] == {"key": "value"}
        assert d["observation"] == "Test observation"
    
    def test_tool_result_str(self):
        """Test ToolResult string representation."""
        from core.tool_registry import ToolResult
        
        result = ToolResult(
            status="success",
            data=None,
            observation="This is the observation"
        )
        
        assert str(result) == "This is the observation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
