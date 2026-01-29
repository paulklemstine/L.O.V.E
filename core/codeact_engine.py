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
"""

import os
import sys
import subprocess
import tempfile
import asyncio
import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Docker Detection
# =============================================================================

class DockerManager:
    """Manages Docker availability detection."""
    
    _docker_available: Optional[bool] = None
    
    @classmethod
    def is_docker_available(cls) -> bool:
        """Check if Docker is available on the system."""
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
            
        logger.info(f"Docker available: {cls._docker_available}")
        return cls._docker_available
    
    @classmethod
    def get_install_instructions(cls) -> str:
        """Get platform-specific Docker installation instructions."""
        if sys.platform == "linux":
            return "curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER"
        elif sys.platform == "darwin":
            return "brew install --cask docker"
        elif sys.platform == "win32":
            return "winget install Docker.DockerDesktop"
        return "Visit https://docs.docker.com/get-docker/"


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
    
    Usage:
        engine = CodeActEngine()
        result = await engine.execute("print('Hello World')")
        
        # With persistent state
        await engine.execute("def greet(name): return f'Hello {name}'")
        result = await engine.execute("print(greet('L.O.V.E.'))")
    """
    
    # Dangerous patterns to block
    BLOCKED_PATTERNS = [
        r"os\s*\.\s*system",
        r"subprocess\s*\.",
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"shutil\s*\.\s*rmtree",
    ]
    
    def __init__(
        self,
        sandbox_mode: str = "subprocess",
        timeout: int = 30,
        llm_runner = None
    ):
        """
        Initialize CodeAct Engine.
        
        Args:
            sandbox_mode: "subprocess" or "docker"
            timeout: Max execution time in seconds
            llm_runner: Async LLM function for self-correction
        """
        self.timeout = timeout
        self.llm_runner = llm_runner
        
        # Auto-detect Docker if requested
        if sandbox_mode == "docker" and not DockerManager.is_docker_available():
            logger.warning("Docker not available, falling back to subprocess")
            sandbox_mode = "subprocess"
        
        self.sandbox_mode = sandbox_mode
        
        # Persistent kernel state (defined functions, variables)
        self.kernel_state: Dict[str, str] = {}
        self.execution_history: List[ThoughtCodeObservation] = []
    
    def _validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """Validate code doesn't contain dangerous patterns."""
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, code):
                return False, f"Dangerous pattern: {pattern}"
        return True, "OK"
    
    def _build_script(self, code: str) -> str:
        """Build execution script with kernel state."""
        preamble = "\n".join(self.kernel_state.values())
        
        return f'''
import sys
import json

# Previous definitions
{preamble}

# Execute new code
try:
    {code}
    print(json.dumps({{"success": True}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e), "type": type(e).__name__}}), file=sys.stderr)
    sys.exit(1)
'''
    
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
            max_retries: Number of self-correction attempts
        """
        # Safety check
        is_safe, reason = self._validate_code_safety(code)
        if not is_safe:
            return CodeExecutionResult(
                success=False, stdout="", stderr=f"Safety: {reason}",
                error_type="SecurityError"
            )
        
        # Execute
        result = await self._execute_subprocess(code)
        
        # Record history
        self.execution_history.append(ThoughtCodeObservation(
            thought=thought or "",
            code=code,
            observation=result.as_observation()
        ))
        
        # Self-correction
        if not result.success and max_retries > 0 and self.llm_runner:
            result = await self._self_correction_loop(code, result, max_retries)
        
        # Extract definitions
        self._extract_definitions(code)
        
        return result
    
    async def _execute_subprocess(self, code: str) -> CodeExecutionResult:
        """Execute code in isolated subprocess."""
        import time
        
        script = self._build_script(code)
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return CodeExecutionResult(
                    success=False, stdout="",
                    stderr=f"Timeout after {self.timeout}s",
                    error_type="TimeoutError"
                )
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            return CodeExecutionResult(
                success=process.returncode == 0,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time=time.time() - start_time,
                error_type="RuntimeError" if process.returncode != 0 else None
            )
            
        except Exception as e:
            return CodeExecutionResult(
                success=False, stdout="", stderr=str(e),
                error_type=type(e).__name__
            )
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
    
    async def _self_correction_loop(
        self,
        original_code: str,
        failed_result: CodeExecutionResult,
        max_retries: int
    ) -> CodeExecutionResult:
        """Attempt to fix failed code using LLM."""
        for attempt in range(max_retries):
            prompt = f"""Fix this Python code that failed:

```python
{original_code}
```

Error: {failed_result.stderr}

Return ONLY the fixed Python code."""
            
            try:
                response = await self.llm_runner(prompt)
                fixed_code = response.get("result", "")
                
                # Extract code from markdown
                if "```python" in fixed_code:
                    match = re.search(r"```python\n(.*?)```", fixed_code, re.DOTALL)
                    if match:
                        fixed_code = match.group(1)
                
                result = await self.execute(fixed_code.strip(), max_retries=0)
                if result.success:
                    return result
                    
            except Exception as e:
                logger.warning(f"Self-correction attempt {attempt + 1} failed: {e}")
        
        return failed_result
    
    def _extract_definitions(self, code: str):
        """Extract function definitions to persist."""
        func_pattern = r"(def\s+(\w+)\s*\([^)]*\):\s*(?:\"\"\".*?\"\"\")?.*?)(?=\ndef\s|\nclass\s|$)"
        
        for match in re.finditer(func_pattern, code, re.DOTALL):
            func_code = match.group(1).strip()
            func_name = match.group(2)
            self.kernel_state[func_name] = func_code
    
    def clear_state(self):
        """Clear all persistent state."""
        self.kernel_state.clear()
        self.execution_history.clear()
    
    def get_defined_functions(self) -> List[str]:
        """Return list of defined function names."""
        return list(self.kernel_state.keys())


# =============================================================================
# Global Instance
# =============================================================================

_codeact_engine: Optional[CodeActEngine] = None


def get_codeact_engine() -> CodeActEngine:
    """Get or create the global CodeActEngine instance."""
    global _codeact_engine
    if _codeact_engine is None:
        _codeact_engine = CodeActEngine()
    return _codeact_engine


async def execute_code(code: str, max_retries: int = 0) -> CodeExecutionResult:
    """Convenience function to execute code."""
    engine = get_codeact_engine()
    return await engine.execute(code, max_retries=max_retries)


def reset_codeact_engine():
    """Reset the global CodeActEngine instance."""
    global _codeact_engine
    if _codeact_engine:
        _codeact_engine.clear_state()
    _codeact_engine = None
