"""
Sandbox Environment - Isolated Execution

Epic 1, Story 1.3: Secure tool validation.

Provides:
1. DockerSandbox: High-isolation container execution (Preferred)
2. LocalSandbox: Process-based fallback (Less secure)
"""

import os
import sys
import subprocess
import threading
import tempfile
import time
from typing import Tuple, List, Optional
from pathlib import Path

from core.logger import log_event


class SandboxError(Exception):
    """Raised when sandbox operations fail."""
    pass


class DockerSandbox:
    """
    Docker-based isolated execution environment.
    
    Story 1.3: Ensures generated tools cannot harm the host system
    or access the network during validation.
    """
    
    def __init__(self, image_name: str = "love_surgeon_sandbox"):
        self.image_name = image_name
        self._check_docker_available()
    
    def _check_docker_available(self):
        """Check if Docker is installed and running."""
        try:
            subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            log_event("⚠️ Docker not available. Sandbox operations will fail.", "WARNING")
    
    def ensure_image_exists(self):
        """Build the sandbox image if it doesn't exist."""
        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                capture_output=True
            )
            
            if result.returncode != 0:
                log_event(f"🏗️ Building sandbox image: {self.image_name}...", "INFO")
                # Build context is core/surgeon/docker
                docker_dir = Path(__file__).parent / "docker"
                if not docker_dir.exists():
                    docker_dir.mkdir(parents=True)
                    # Create default Dockerfile if missing
                    self._create_default_dockerfile(docker_dir)
                
                subprocess.run(
                    ["docker", "build", "-t", self.image_name, "."],
                    cwd=str(docker_dir),
                    check=True,
                    capture_output=True
                )
                log_event("✅ Sandbox image built successfully", "INFO")
                
        except Exception as e:
            log_event(f"❌ Failed to build sandbox image: {e}", "ERROR")
            raise SandboxError(f"Docker build failed: {e}")
    
    def _create_default_dockerfile(self, docker_dir: Path):
        """Create a default Dockerfile if none exists."""
        dockerfile_content = """
FROM python:3.12-slim

# Install minimal dependencies for testing
RUN pip install pytest radon

# Create non-root user for security
RUN useradd -m sandbox
USER sandbox

WORKDIR /project

# Set default command
CMD ["python3", "--version"]
"""
        with open(docker_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content.strip())
    
    def run_command(
        self, 
        command: str, 
        timeout: int = 60, 
        network_disabled: bool = True,
        mounts: List[Tuple[str, str]] = None
    ) -> Tuple[int, str, str]:
        """
        Run a command inside the sandbox container.
        
        Args:
            command: Shell command to run
            timeout: Timeout in seconds
            network_disabled: If True, runs with --network none
            mounts: List of (host_path, container_path) tuples
            
        Returns:
            (exit_code, stdout, stderr)
        """
        cmd = [
            "docker", "run", "--rm",
            "--memory=512m",  # Limit memory
            "--cpus=1.0",     # Limit CPU
        ]
        
        if network_disabled:
            cmd.extend(["--network", "none"])
        
        if mounts:
            for host_path, container_path in mounts:
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        cmd.extend([self.image_name, "bash", "-c", command])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            log_event(f"⏱️ Sandbox command timed out after {timeout}s", "WARNING")
            return 124, "", "Operation timed out"
            
        except Exception as e:
            log_event(f"❌ Sandbox execution error: {e}", "ERROR")
            return 1, "", str(e)


class LocalSandbox:
    """
    Fallback subprocess-based execution (Less Secure).
    
    Used when Docker is not available. Provides basic timeout protection
    but NO isolation (file system, network, etc).
    """
    
    def __init__(self):
        log_event("⚠️ Using LocalSandbox (Unsecured Fallback)", "WARNING")
    
    def ensure_image_exists(self):
        """No-op for local sandbox."""
        pass
    
    def run_command(
        self, 
        command: str, 
        timeout: int = 60, 
        network_disabled: bool = False, # Cannot enforce locally
        mounts: List[Tuple[str, str]] = None
    ) -> Tuple[int, str, str]:
        """
        Run command in local subprocess.
        
        Args:
            command: Shell command to run
            timeout: Timeout in seconds
            
        Returns:
            (exit_code, stdout, stderr)
        """
        # Prepare environment
        env = os.environ.copy()
        
        # If mounts passed (e.g. for creating temp files), we might need to 
        # ensure the command looks at the right place. 
        # For simplicity in local mode, we assume paths are already local.
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=mounts[0][0] if mounts else None # Run in project root if mounted
            )
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return 124, "", "Operation timed out"
            
        except Exception as e:
            return 1, "", str(e)


def get_sandbox(prefer_docker: bool = True):
    """Factory to get the appropriate sandbox."""
    if prefer_docker:
        try:
            # Check if docker is actually usable
            subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                check=True
            )
            return DockerSandbox()
        except Exception:
            pass
            
    return LocalSandbox()
