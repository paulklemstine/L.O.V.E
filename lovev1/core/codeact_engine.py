"""
CodeAct Engine: Code-as-Action Paradigm for Dynamic Tool Generation

This module implements the "LLM as Engineer" pattern where agents write
executable Python code as their actions, rather than selecting from
predefined tools.

Key Features:
- Thought-Code-Observation loop for iterative refinement
- Define-and-Use pattern for persistent function definitions
- Self-correction with error feedback
- Subprocess sandbox with Docker fallback

References:
- CodeAct paradigm (University of Illinois / OpenHands)
- Design doc: "The Architectural Evolution of Open Source AI"
"""

import os
import sys
import subprocess
import tempfile
import shutil
import asyncio
import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from core.logging import log_event

logger = logging.getLogger(__name__)


# =============================================================================
# Docker Detection and Installation
# =============================================================================

class DockerManager:
    """Manages Docker availability detection and installation."""
    
    _docker_available: Optional[bool] = None
    
    @classmethod
    def is_docker_available(cls) -> bool:
        """
        Check if Docker is available on the system.
        
        Returns:
            True if Docker is installed and running
        """
        if cls._docker_available is not None:
            return cls._docker_available
            
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            cls._docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            cls._docker_available = False
            
        log_event("codeact", f"Docker available: {cls._docker_available}")
        return cls._docker_available
    
    @classmethod
    def get_docker_install_instructions(cls) -> str:
        """Get platform-specific Docker installation instructions."""
        if sys.platform == "linux":
            return """
Docker Installation (Linux):
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER
  # Log out and back in for group changes
"""
        elif sys.platform == "darwin":
            return """
Docker Installation (macOS):
  brew install --cask docker
  # Or download from https://www.docker.com/products/docker-desktop
"""
        elif sys.platform == "win32":
            return """
Docker Installation (Windows):
  winget install Docker.DockerDesktop
  # Or download from https://www.docker.com/products/docker-desktop
"""
        return "Visit https://docs.docker.com/get-docker/"
    
    @classmethod
    async def attempt_docker_install(cls) -> Tuple[bool, str]:
        """
        Attempt to install Docker automatically (Linux only).
        
        Returns:
            Tuple of (success, message)
        """
        if sys.platform != "linux":
            return False, f"Automatic Docker installation only supported on Linux. {cls.get_docker_install_instructions()}"
        
        # Check if we have sudo access
        try:
            result = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True
            )
            if result.returncode != 0:
                return False, "Docker installation requires sudo. Run manually:\n" + cls.get_docker_install_instructions()
        except FileNotFoundError:
            return False, "sudo not available. " + cls.get_docker_install_instructions()
        
        log_event("codeact", "Attempting automatic Docker installation...")
        
        try:
            # Download Docker install script
            download = subprocess.run(
                ["curl", "-fsSL", "https://get.docker.com", "-o", "/tmp/get-docker.sh"],
                capture_output=True,
                timeout=60
            )
            if download.returncode != 0:
                return False, "Failed to download Docker installer"
            
            # Run install script
            install = subprocess.run(
                ["sudo", "sh", "/tmp/get-docker.sh"],
                capture_output=True,
                timeout=300
            )
            if install.returncode != 0:
                return False, f"Docker install failed: {install.stderr.decode()}"
            
            # Add current user to docker group
            subprocess.run(
                ["sudo", "usermod", "-aG", "docker", os.environ.get("USER", "")],
                capture_output=True
            )
            
            # Start Docker service
            subprocess.run(
                ["sudo", "systemctl", "start", "docker"],
                capture_output=True
            )
            
            # Reset cache and verify
            cls._docker_available = None
            if cls.is_docker_available():
                return True, "Docker installed successfully!"
            else:
                return True, "Docker installed. Please log out and back in for group permissions."
                
        except subprocess.TimeoutExpired:
            return False, "Docker installation timed out"
        except Exception as e:
            return False, f"Docker installation failed: {e}"


# =============================================================================
# Execution Results
# =============================================================================

@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    return_value: Any = None
    execution_time: float = 0.0
    error_type: Optional[str] = None
    
    def as_observation(self) -> str:
        """Format result as LLM observation."""
        if self.success:
            output = self.stdout.strip() if self.stdout else "(no output)"
            return f"[Execution SUCCESS]\nOutput:\n{output}"
        else:
            return f"[Execution FAILED]\nError ({self.error_type}):\n{self.stderr}"


@dataclass
class ThoughtCodeObservation:
    """A single iteration of the TCO loop."""
    thought: str
    code: str
    observation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# CodeAct Engine
# =============================================================================

class CodeActEngine:
    """
    Code-as-Action paradigm implementation.
    
    The agent writes Python code as its "action" rather than selecting
    from predefined tools. This enables infinite tool capability.
    
    Architecture:
    - Subprocess sandbox (default): Fast, isolated Python subprocess
    - Docker sandbox (optional): Full container isolation
    
    Usage:
        engine = CodeActEngine()
        result = await engine.execute("print('Hello World')")
        
        # With persistent state
        await engine.execute("def greet(name): return f'Hello {name}'")
        result = await engine.execute("greet('L.O.V.E.')")
    """
    
    # Dangerous imports that should be blocked in sandbox
    BLOCKED_IMPORTS = {
        "os.system", "subprocess", "shutil.rmtree",
        "socket", "urllib", "requests", "httpx",
        "__import__", "eval", "exec", "compile"
    }
    
    # Safe subset of imports allowed
    SAFE_IMPORTS = {
        "math", "json", "datetime", "re", "collections",
        "itertools", "functools", "operator", "string",
        "random", "statistics", "decimal", "fractions"
    }
    
    def __init__(
        self,
        sandbox_mode: str = "subprocess",
        timeout: int = 30,
        memory_limit_mb: int = 256,
        network_allowed: bool = False,
        llm_runner = None
    ):
        """
        Initialize CodeAct Engine.
        
        Args:
            sandbox_mode: "subprocess" or "docker"
            timeout: Max execution time in seconds
            memory_limit_mb: Memory limit for Docker mode
            network_allowed: Whether network access is permitted
            llm_runner: Async LLM function for self-correction
        """
        self.sandbox_mode = sandbox_mode
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.network_allowed = network_allowed
        self.llm_runner = llm_runner
        
        # Persistent kernel state (defined functions, variables)
        self.kernel_state: Dict[str, str] = {}
        self.execution_history: List[ThoughtCodeObservation] = []
        
        # Auto-detect Docker if requested
        if sandbox_mode == "docker" and not DockerManager.is_docker_available():
            logger.warning("Docker requested but not available. Falling back to subprocess.")
            self.sandbox_mode = "subprocess"
    
    def _get_llm_runner(self):
        """Lazy-load LLM runner."""
        if self.llm_runner is None:
            from core.llm_api import run_llm
            self.llm_runner = run_llm
        return self.llm_runner
    
    def _validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """
        Validate code doesn't contain dangerous patterns.
        
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check for blocked imports
        for blocked in self.BLOCKED_IMPORTS:
            if blocked in code:
                return False, f"Blocked import/function: {blocked}"
        
        # Check for shell execution attempts
        dangerous_patterns = [
            r"os\s*\.\s*system",
            r"subprocess\s*\.",
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\([^)]*['\"][wa]['\"]",  # Write mode files
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, "Code passed safety validation"
    
    def _build_execution_script(self, code: str) -> str:
        """Build a complete Python script with kernel state."""
        # Combine all defined functions/variables with new code
        preamble = "\n".join(self.kernel_state.values())
        
        script = f'''
import sys
import json

# Previous definitions
{preamble}

# New code to execute
try:
    _result = None
    {code}
    
    # Try to capture last expression result
    print(json.dumps({{"success": True, "result": str(_result) if _result else None}}))
except Exception as e:
    print(json.dumps({{"success": False, "error_type": type(e).__name__, "error": str(e)}}), file=sys.stderr)
    sys.exit(1)
'''
        return script
    
    async def execute(
        self,
        code: str,
        thought: Optional[str] = None,
        max_retries: int = 0
    ) -> CodeExecutionResult:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            thought: Optional reasoning behind this code
            max_retries: Number of self-correction attempts (0 = no retry)
            
        Returns:
            CodeExecutionResult with stdout/stderr and success status
        """
        # Safety check
        is_safe, reason = self._validate_code_safety(code)
        if not is_safe:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=f"Safety validation failed: {reason}",
                error_type="SecurityError"
            )
        
        # Execute based on sandbox mode
        if self.sandbox_mode == "docker":
            result = await self._execute_docker(code)
        else:
            result = await self._execute_subprocess(code)
        
        # Record in history
        self.execution_history.append(ThoughtCodeObservation(
            thought=thought or "",
            code=code,
            observation=result.as_observation()
        ))
        
        # Self-correction loop
        if not result.success and max_retries > 0:
            result = await self._self_correction_loop(code, result, max_retries)
        
        # If code defines a function, add to kernel state
        self._extract_definitions(code)
        
        log_event("codeact", f"Executed code: success={result.success}")
        return result
    
    async def _execute_subprocess(self, code: str) -> CodeExecutionResult:
        """Execute code in isolated subprocess."""
        import time
        
        script = self._build_execution_script(code)
        start_time = time.time()
        
        try:
            # Create temp file for script
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(script)
                script_path = f.name
            
            # Execute in subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return CodeExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout}s",
                    error_type="TimeoutError",
                    execution_time=time.time() - start_time
                )
            
            execution_time = time.time() - start_time
            
            # Parse result
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            if process.returncode == 0:
                return CodeExecutionResult(
                    success=True,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    execution_time=execution_time
                )
            else:
                # Try to parse error type from stderr
                error_type = "RuntimeError"
                try:
                    error_data = json.loads(stderr_str)
                    error_type = error_data.get("error_type", "RuntimeError")
                except:
                    pass
                    
                return CodeExecutionResult(
                    success=False,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    error_type=error_type,
                    execution_time=execution_time
                )
                
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                error_type=type(e).__name__
            )
        finally:
            # Cleanup
            try:
                os.unlink(script_path)
            except:
                pass
    
    async def _execute_docker(self, code: str) -> CodeExecutionResult:
        """Execute code in Docker container."""
        import time
        
        if not DockerManager.is_docker_available():
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr="Docker not available. " + DockerManager.get_docker_install_instructions(),
                error_type="DockerNotAvailable"
            )
        
        script = self._build_execution_script(code)
        start_time = time.time()
        
        try:
            # Create temp directory for script
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = os.path.join(tmpdir, "script.py")
                with open(script_path, 'w') as f:
                    f.write(script)
                
                # Build docker command
                network_mode = "none" if not self.network_allowed else "bridge"
                docker_cmd = [
                    "docker", "run",
                    "--rm",
                    f"--network={network_mode}",
                    f"--memory={self.memory_limit_mb}m",
                    "--cpus=1",
                    "--read-only",
                    f"--volume={tmpdir}:/code:ro",
                    "--workdir=/code",
                    "python:3.11-slim",
                    "python", "/code/script.py"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout + 10  # Extra time for container startup
                    )
                except asyncio.TimeoutError:
                    # Kill container
                    subprocess.run(["docker", "kill", process.pid], capture_output=True)
                    return CodeExecutionResult(
                        success=False,
                        stdout="",
                        stderr=f"Docker execution timed out after {self.timeout}s",
                        error_type="TimeoutError"
                    )
                
                execution_time = time.time() - start_time
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')
                
                return CodeExecutionResult(
                    success=process.returncode == 0,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    execution_time=execution_time,
                    error_type="RuntimeError" if process.returncode != 0 else None
                )
                
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                error_type=type(e).__name__
            )
    
    async def _self_correction_loop(
        self,
        original_code: str,
        failed_result: CodeExecutionResult,
        max_retries: int
    ) -> CodeExecutionResult:
        """
        Attempt to fix failed code using LLM.
        
        Implements the self-correction pattern from the design doc.
        """
        llm_runner = self._get_llm_runner()
        
        for attempt in range(max_retries):
            prompt = f"""The following Python code failed to execute:

```python
{original_code}
```

Error ({failed_result.error_type}):
{failed_result.stderr}

Please fix the code. Return ONLY the corrected Python code, no explanations.
"""
            
            try:
                response = await llm_runner(prompt)
                fixed_code = response.get("result", "")
                
                # Extract code from markdown if present
                if "```python" in fixed_code:
                    match = re.search(r"```python\n(.*?)```", fixed_code, re.DOTALL)
                    if match:
                        fixed_code = match.group(1)
                elif "```" in fixed_code:
                    match = re.search(r"```\n?(.*?)```", fixed_code, re.DOTALL)
                    if match:
                        fixed_code = match.group(1)
                
                # Try the fixed code
                result = await self.execute(fixed_code.strip(), max_retries=0)
                
                if result.success:
                    log_event("codeact", f"Self-correction succeeded on attempt {attempt + 1}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Self-correction attempt {attempt + 1} failed: {e}")
        
        return failed_result
    
    def _extract_definitions(self, code: str):
        """Extract function/class definitions to persist in kernel state."""
        # Match function definitions
        func_pattern = r"(def\s+(\w+)\s*\([^)]*\):\s*(?:\"\"\".*?\"\"\")?.*?)(?=\ndef\s|\nclass\s|$)"
        
        for match in re.finditer(func_pattern, code, re.DOTALL):
            func_code = match.group(1).strip()
            func_name = match.group(2)
            self.kernel_state[func_name] = func_code
            log_event("codeact", f"Persisted function: {func_name}")
        
        # Match class definitions
        class_pattern = r"(class\s+(\w+).*?(?=\nclass\s|\ndef\s|$))"
        
        for match in re.finditer(class_pattern, code, re.DOTALL):
            class_code = match.group(1).strip()
            class_name = match.group(2)
            self.kernel_state[class_name] = class_code
            log_event("codeact", f"Persisted class: {class_name}")
    
    def clear_state(self):
        """Clear all persistent kernel state."""
        self.kernel_state.clear()
        self.execution_history.clear()
    
    def get_defined_functions(self) -> List[str]:
        """Return list of currently defined function names."""
        return list(self.kernel_state.keys())


# =============================================================================
# Global Instance & Convenience Functions
# =============================================================================

_codeact_engine: Optional[CodeActEngine] = None


def get_codeact_engine() -> CodeActEngine:
    """Get or create the global CodeActEngine instance."""
    global _codeact_engine
    if _codeact_engine is None:
        _codeact_engine = CodeActEngine()
    return _codeact_engine


async def execute_code(code: str, max_retries: int = 0) -> CodeExecutionResult:
    """
    Convenience function to execute code.
    
    Args:
        code: Python code to execute
        max_retries: Number of self-correction attempts
        
    Returns:
        CodeExecutionResult
    """
    engine = get_codeact_engine()
    return await engine.execute(code, max_retries=max_retries)


def reset_codeact_engine():
    """Reset the global CodeActEngine instance."""
    global _codeact_engine
    if _codeact_engine:
        _codeact_engine.clear_state()
    _codeact_engine = None
