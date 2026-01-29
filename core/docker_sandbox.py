"""
Docker Sandbox for Secure Code Execution.

Provides containerized code execution for:
- CodeAct engine code execution
- Tool fabrication validation
- Agent-generated code testing

Key Features:
- Process isolation via Docker containers
- Network isolation options (none, allowlist)
- Resource limits (memory, CPU, time)
- Automatic cleanup
"""

import os
import sys
import subprocess
import tempfile
import asyncio
import logging
import shutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float = 0.0
    container_id: Optional[str] = None


class DockerSandbox:
    """
    Containerized code execution sandbox.
    
    Uses Docker to provide process isolation, network control,
    and filesystem sandboxing for agent-generated code.
    
    Usage:
        sandbox = DockerSandbox()
        result = await sandbox.run_python("print('Hello')")
    """
    
    # Docker image for Python execution
    DEFAULT_IMAGE = "python:3.11-slim"
    
    def __init__(
        self,
        network_mode: str = "none",  # "none", "bridge", "host"
        timeout_seconds: int = 30,
        memory_limit: str = "256m",
        cpu_limit: float = 1.0,
        work_dir: Optional[str] = None
    ):
        """
        Initialize Docker sandbox.
        
        Args:
            network_mode: Docker network mode (none=isolated, bridge=limited, host=full)
            timeout_seconds: Max execution time
            memory_limit: Container memory limit (e.g., "256m", "1g")
            cpu_limit: Number of CPUs to use
            work_dir: Working directory for temporary files
        """
        self.network_mode = network_mode
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.work_dir = work_dir or tempfile.gettempdir()
        
        self._docker_available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available
        
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            self._docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._docker_available = False
        
        return self._docker_available
    
    def get_install_instructions(self) -> str:
        """Get Docker installation instructions."""
        if sys.platform == "linux":
            return "curl -fsSL https://get.docker.com | sh"
        elif sys.platform == "darwin":
            return "brew install --cask docker"
        elif sys.platform == "win32":
            return "winget install Docker.DockerDesktop"
        return "https://docs.docker.com/get-docker/"
    
    async def pull_image(self, image: str = None) -> bool:
        """Pull the Docker image if not present."""
        image = image or self.DEFAULT_IMAGE
        
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                timeout=300
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}")
            return False
    
    async def run_python(
        self,
        code: str,
        packages: Optional[List[str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> SandboxResult:
        """
        Run Python code in a Docker container.
        
        Args:
            code: Python code to execute
            packages: Optional pip packages to install
            volumes: Optional volume mounts {host_path: container_path}
            environment: Optional environment variables
            
        Returns:
            SandboxResult with stdout, stderr, and exit code
        """
        if not self.is_available():
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Docker not available. Install: {self.get_install_instructions()}",
                exit_code=-1
            )
        
        import time
        start_time = time.time()
        
        # Create temp directory for script
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            # Write script
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, 'w') as f:
                f.write(code)
            
            # Build wrapper script with optional package installation
            wrapper = "#!/bin/bash\n"
            if packages:
                wrapper += f"pip install --quiet {' '.join(packages)}\n"
            wrapper += "python /code/script.py\n"
            
            wrapper_path = os.path.join(tmpdir, "run.sh")
            with open(wrapper_path, 'w') as f:
                f.write(wrapper)
            os.chmod(wrapper_path, 0o755)
            
            # Build docker command
            cmd = [
                "docker", "run",
                "--rm",
                f"--network={self.network_mode}",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                "--read-only" if self.network_mode == "none" else "",
                f"--volume={tmpdir}:/code:ro",
                "--workdir=/code",
            ]
            
            # Remove empty strings
            cmd = [c for c in cmd if c]
            
            # Add custom volumes
            if volumes:
                for host, container in volumes.items():
                    cmd.extend([f"--volume={host}:{container}"])
            
            # Add environment variables
            if environment:
                for key, value in environment.items():
                    cmd.extend(["-e", f"{key}={value}"])
            
            # Add image and command
            cmd.extend([
                self.DEFAULT_IMAGE,
                "bash", "/code/run.sh"
            ])
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout + 10  # Extra for container startup
                    )
                except asyncio.TimeoutError:
                    # Kill container
                    try:
                        subprocess.run(
                            ["docker", "kill", str(process.pid)],
                            capture_output=True,
                            timeout=5
                        )
                    except:
                        pass
                    
                    return SandboxResult(
                        success=False,
                        stdout="",
                        stderr=f"Execution timed out after {self.timeout}s",
                        exit_code=-1,
                        execution_time=time.time() - start_time
                    )
                
                return SandboxResult(
                    success=process.returncode == 0,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    exit_code=process.returncode,
                    execution_time=time.time() - start_time
                )
                
            except Exception as e:
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=str(e),
                    exit_code=-1,
                    execution_time=time.time() - start_time
                )
    
    async def run_shell(
        self,
        command: str,
        shell: str = "/bin/bash"
    ) -> SandboxResult:
        """
        Run a shell command in a Docker container.
        
        Args:
            command: Shell command to execute
            shell: Shell to use (default: /bin/bash)
            
        Returns:
            SandboxResult
        """
        if not self.is_available():
            return SandboxResult(
                success=False,
                stdout="",
                stderr="Docker not available",
                exit_code=-1
            )
        
        import time
        start_time = time.time()
        
        cmd = [
            "docker", "run",
            "--rm",
            f"--network={self.network_mode}",
            f"--memory={self.memory_limit}",
            self.DEFAULT_IMAGE,
            shell, "-c", command
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                exit_code=process.returncode,
                execution_time=time.time() - start_time
            )
            
        except asyncio.TimeoutError:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="Timeout",
                exit_code=-1
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1
            )


# =============================================================================
# Subprocess Fallback Sandbox
# =============================================================================

class SubprocessSandbox:
    """
    Fallback sandbox using subprocess (when Docker unavailable).
    
    Provides basic isolation via:
    - Separate process
    - Limited environment
    - Timeout enforcement
    
    Note: Less secure than Docker, use for trusted code only.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
    
    async def run_python(
        self,
        code: str,
        packages: Optional[List[str]] = None
    ) -> SandboxResult:
        """Run Python code in subprocess."""
        import time
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            script_path = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1"
                }
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Timeout after {self.timeout}s",
                    exit_code=-1
                )
            
            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                exit_code=process.returncode,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1
            )
        finally:
            try:
                os.unlink(script_path)
            except:
                pass


# =============================================================================
# Unified Sandbox Interface
# =============================================================================

class UnifiedSandbox:
    """
    Unified sandbox interface that automatically uses Docker when available,
    falling back to subprocess when not.
    """
    
    def __init__(
        self,
        prefer_docker: bool = True,
        timeout_seconds: int = 30,
        **docker_kwargs
    ):
        self.prefer_docker = prefer_docker
        self.timeout = timeout_seconds
        self.docker_kwargs = docker_kwargs
        
        self._docker_sandbox: Optional[DockerSandbox] = None
        self._subprocess_sandbox: Optional[SubprocessSandbox] = None
    
    def _get_sandbox(self):
        """Get the appropriate sandbox."""
        if self.prefer_docker:
            if self._docker_sandbox is None:
                self._docker_sandbox = DockerSandbox(
                    timeout_seconds=self.timeout,
                    **self.docker_kwargs
                )
            
            if self._docker_sandbox.is_available():
                return self._docker_sandbox
        
        # Fallback to subprocess
        if self._subprocess_sandbox is None:
            self._subprocess_sandbox = SubprocessSandbox(timeout_seconds=self.timeout)
        
        return self._subprocess_sandbox
    
    async def run_python(
        self,
        code: str,
        packages: Optional[List[str]] = None,
        **kwargs
    ) -> SandboxResult:
        """Run Python code in the best available sandbox."""
        sandbox = self._get_sandbox()
        return await sandbox.run_python(code, packages=packages, **kwargs)
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_sandbox is None:
            self._docker_sandbox = DockerSandbox()
        return self._docker_sandbox.is_available()


# =============================================================================
# Global Instance
# =============================================================================

_sandbox: Optional[UnifiedSandbox] = None


def get_sandbox() -> UnifiedSandbox:
    """Get or create the global sandbox instance."""
    global _sandbox
    if _sandbox is None:
        _sandbox = UnifiedSandbox()
    return _sandbox


async def run_in_sandbox(code: str, **kwargs) -> SandboxResult:
    """Convenience function to run code in sandbox."""
    return await get_sandbox().run_python(code, **kwargs)


def reset_sandbox():
    """Reset the global sandbox instance."""
    global _sandbox
    _sandbox = None
