
import pytest
from unittest.mock import patch
from core.surgeon.sandbox import get_sandbox, DockerSandbox, LocalSandbox

class TestSandboxFactory:
    """Tests for the sandbox factory."""

    @patch("core.surgeon.sandbox.is_docker_available")
    def test_get_sandbox_docker(self, mock_is_docker_available):
        """Test that DockerSandbox is returned when Docker is available."""
        mock_is_docker_available.return_value = True
        sandbox = get_sandbox()
        assert isinstance(sandbox, DockerSandbox)

    @patch("core.surgeon.sandbox.is_docker_available")
    def test_get_sandbox_local(self, mock_is_docker_available):
        """Test that LocalSandbox is returned when Docker is not available."""
        mock_is_docker_available.return_value = False
        sandbox = get_sandbox()
        assert isinstance(sandbox, LocalSandbox)


class TestLocalSandbox:
    """Tests for the LocalSandbox."""

    def test_run_command_success(self):
        """Test that a simple command runs successfully."""
        sandbox = LocalSandbox()
        return_code, stdout, stderr = sandbox.run_command("echo 'hello world'")
        assert return_code == 0
        assert "hello world" in stdout

    def test_run_command_failure(self):
        """Test that a failing command returns a non-zero exit code."""
        sandbox = LocalSandbox()
        return_code, stdout, stderr = sandbox.run_command("exit 1")
        assert return_code == 1

    def test_run_command_timeout(self):
        """Test that a command times out."""
        sandbox = LocalSandbox()
        return_code, stdout, stderr = sandbox.run_command("sleep 5", timeout=1)
        assert return_code == -1
        assert "timed out" in stderr.lower()

    def test_run_command_scratch_dir(self, tmp_path):
        """Test that the command is run in the scratch directory."""
        sandbox = LocalSandbox(scratch_dir=str(tmp_path))
        return_code, stdout, stderr = sandbox.run_command("pwd")
        assert return_code == 0
        assert str(tmp_path) in stdout
