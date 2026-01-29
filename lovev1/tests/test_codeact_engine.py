"""
Tests for CodeAct Engine: Code-as-Action Paradigm

Tests the dynamic code execution capabilities including:
- Subprocess sandbox execution
- Docker sandbox (if available)
- Self-correction loops
- Kernel state persistence (define-and-use pattern)
- Safety validation
"""

import pytest
import asyncio
import sys
from unittest.mock import patch, MagicMock, AsyncMock

from core.codeact_engine import (
    CodeActEngine,
    DockerManager,
    CodeExecutionResult,
    ThoughtCodeObservation,
    get_codeact_engine,
    execute_code,
    reset_codeact_engine
)


# =============================================================================
# DockerManager Tests
# =============================================================================

class TestDockerManager:
    """Tests for Docker detection and installation helpers."""
    
    def setup_method(self):
        """Reset Docker state before each test."""
        DockerManager._docker_available = None
    
    def test_is_docker_available_detects_docker(self):
        """Should detect if Docker is running."""
        # This test's result depends on the actual environment
        result = DockerManager.is_docker_available()
        assert isinstance(result, bool)
    
    def test_docker_availability_is_cached(self):
        """Should cache Docker availability check."""
        DockerManager._docker_available = True
        assert DockerManager.is_docker_available() is True
        
        DockerManager._docker_available = False
        assert DockerManager.is_docker_available() is False
    
    def test_get_docker_install_instructions(self):
        """Should return platform-specific install instructions."""
        instructions = DockerManager.get_docker_install_instructions()
        assert isinstance(instructions, str)
        assert "docker" in instructions.lower() or "Docker" in instructions
    
    @pytest.mark.asyncio
    async def test_attempt_docker_install_non_linux(self):
        """Should return helpful message on non-Linux platforms."""
        with patch('core.codeact_engine.sys.platform', 'darwin'):
            success, message = await DockerManager.attempt_docker_install()
            assert success is False
            assert "Linux" in message or "install" in message.lower()


# =============================================================================
# CodeActEngine Basic Tests
# =============================================================================

class TestCodeActEngineBasic:
    """Basic tests for CodeActEngine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create a fresh CodeActEngine for each test."""
        return CodeActEngine(sandbox_mode="subprocess", timeout=10)
    
    def test_engine_initialization(self, engine):
        """Should initialize with correct defaults."""
        assert engine.sandbox_mode == "subprocess"
        assert engine.timeout == 10
        assert engine.kernel_state == {}
        assert engine.execution_history == []
    
    def test_engine_fallback_from_docker(self):
        """Should fall back to subprocess if Docker unavailable."""
        with patch.object(DockerManager, 'is_docker_available', return_value=False):
            engine = CodeActEngine(sandbox_mode="docker")
            assert engine.sandbox_mode == "subprocess"


# =============================================================================
# Code Safety Validation Tests
# =============================================================================

class TestCodeSafetyValidation:
    """Tests for code safety validation."""
    
    @pytest.fixture
    def engine(self):
        return CodeActEngine()
    
    def test_safe_code_passes(self, engine):
        """Safe code should pass validation."""
        is_safe, _ = engine._validate_code_safety("x = 1 + 1\nprint(x)")
        assert is_safe is True
    
    def test_os_system_blocked(self, engine):
        """os.system calls should be blocked."""
        is_safe, reason = engine._validate_code_safety("import os; os.system('rm -rf /')")
        assert is_safe is False
        assert "os.system" in reason.lower() or "blocked" in reason.lower()
    
    def test_subprocess_blocked(self, engine):
        """subprocess module should be blocked."""
        is_safe, reason = engine._validate_code_safety("import subprocess; subprocess.run(['ls'])")
        assert is_safe is False
    
    def test_eval_blocked(self, engine):
        """eval should be blocked."""
        is_safe, reason = engine._validate_code_safety("eval('print(1)')")
        assert is_safe is False
    
    def test_exec_blocked(self, engine):
        """exec should be blocked."""
        is_safe, reason = engine._validate_code_safety("exec('x=1')")
        assert is_safe is False


# =============================================================================
# Subprocess Execution Tests
# =============================================================================

class TestSubprocessExecution:
    """Tests for subprocess sandbox execution."""
    
    @pytest.fixture
    def engine(self):
        return CodeActEngine(sandbox_mode="subprocess", timeout=10)
    
    @pytest.mark.asyncio
    async def test_simple_print(self, engine):
        """Should execute simple print statement."""
        result = await engine.execute("print('Hello World')")
        assert result.success is True
        assert "Hello World" in result.stdout
    
    @pytest.mark.asyncio
    async def test_arithmetic(self, engine):
        """Should execute arithmetic operations."""
        result = await engine.execute("print(2 + 2)")
        assert result.success is True
        assert "4" in result.stdout
    
    @pytest.mark.asyncio
    async def test_syntax_error(self, engine):
        """Should handle syntax errors gracefully."""
        result = await engine.execute("print('unclosed")
        assert result.success is False
        assert result.error_type is not None
    
    @pytest.mark.asyncio
    async def test_runtime_error(self, engine):
        """Should handle runtime errors gracefully."""
        result = await engine.execute("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in result.error_type or "error" in result.stderr.lower()
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Should timeout on long-running code."""
        engine = CodeActEngine(timeout=1)
        result = await engine.execute("import time; time.sleep(10)")
        assert result.success is False
        assert "timeout" in result.stderr.lower()
    
    @pytest.mark.asyncio
    async def test_blocked_code_rejected(self, engine):
        """Should reject blocked code without execution."""
        result = await engine.execute("import os; os.system('ls')")
        assert result.success is False
        assert "security" in result.error_type.lower() or "safety" in result.stderr.lower()


# =============================================================================
# Define-and-Use Pattern Tests
# =============================================================================

class TestDefineAndUsePattern:
    """Tests for persistent function definitions (kernel state)."""
    
    @pytest.fixture
    def engine(self):
        return CodeActEngine(sandbox_mode="subprocess", timeout=10)
    
    @pytest.mark.asyncio
    async def test_define_then_use_function(self, engine):
        """Should be able to define a function and use it later."""
        # Define a function
        define_result = await engine.execute("""
def greet(name):
    return f"Hello, {name}!"
print("Function defined")
""")
        assert define_result.success is True
        assert "greet" in engine.kernel_state
        
        # Use the function
        use_result = await engine.execute("print(greet('World'))")
        assert use_result.success is True
        assert "Hello, World!" in use_result.stdout
    
    @pytest.mark.asyncio
    async def test_multiple_functions(self, engine):
        """Should persist multiple function definitions."""
        await engine.execute("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        
        assert "add" in engine.kernel_state
        assert "multiply" in engine.kernel_state
        
        result = await engine.execute("print(add(2, 3), multiply(2, 3))")
        assert result.success is True
        assert "5" in result.stdout
        assert "6" in result.stdout
    
    def test_get_defined_functions(self, engine):
        """Should list defined functions."""
        engine.kernel_state = {"func1": "def func1(): pass", "func2": "def func2(): pass"}
        funcs = engine.get_defined_functions()
        assert "func1" in funcs
        assert "func2" in funcs
    
    def test_clear_state(self, engine):
        """Should clear all kernel state."""
        engine.kernel_state = {"test": "def test(): pass"}
        engine.execution_history = [ThoughtCodeObservation("t", "c", "o")]
        
        engine.clear_state()
        
        assert engine.kernel_state == {}
        assert engine.execution_history == []


# =============================================================================
# Self-Correction Loop Tests
# =============================================================================

class TestSelfCorrectionLoop:
    """Tests for the self-correction capability."""
    
    @pytest.fixture
    def engine(self):
        engine = CodeActEngine(sandbox_mode="subprocess", timeout=10)
        return engine
    
    @pytest.mark.asyncio
    async def test_self_correction_fixes_error(self, engine):
        """Should attempt to fix errors using LLM."""
        # Mock LLM to return fixed code
        async def mock_llm(prompt):
            return {"result": "print('fixed code')"}
        
        engine.llm_runner = mock_llm
        
        # This code has an error, should trigger self-correction
        result = await engine.execute("prnt('typo')", max_retries=1)
        
        # The correction should have been attempted
        assert len(engine.execution_history) >= 1


# =============================================================================
# Execution Result Tests
# =============================================================================

class TestCodeExecutionResult:
    """Tests for CodeExecutionResult dataclass."""
    
    def test_success_observation(self):
        """Should format successful execution as observation."""
        result = CodeExecutionResult(
            success=True,
            stdout="Hello World\n",
            stderr=""
        )
        obs = result.as_observation()
        assert "SUCCESS" in obs
        assert "Hello World" in obs
    
    def test_failure_observation(self):
        """Should format failed execution as observation."""
        result = CodeExecutionResult(
            success=False,
            stdout="",
            stderr="NameError: name 'x' is not defined",
            error_type="NameError"
        )
        obs = result.as_observation()
        assert "FAILED" in obs
        assert "NameError" in obs


# =============================================================================
# Global Instance Tests
# =============================================================================

class TestGlobalInstance:
    """Tests for global engine instance."""
    
    def setup_method(self):
        reset_codeact_engine()
    
    def test_get_codeact_engine_singleton(self):
        """Should return same instance."""
        engine1 = get_codeact_engine()
        engine2 = get_codeact_engine()
        assert engine1 is engine2
    
    @pytest.mark.asyncio
    async def test_execute_code_convenience(self):
        """Should execute code via convenience function."""
        reset_codeact_engine()
        result = await execute_code("print('test')")
        assert result.success is True


# =============================================================================
# Docker Execution Tests (Conditional)
# =============================================================================

@pytest.mark.skipif(
    not DockerManager.is_docker_available(),
    reason="Docker not available"
)
class TestDockerExecution:
    """Tests for Docker sandbox execution (only run if Docker available)."""
    
    @pytest.fixture
    def engine(self):
        return CodeActEngine(sandbox_mode="docker", timeout=30)
    
    @pytest.mark.asyncio
    async def test_docker_simple_print(self, engine):
        """Should execute in Docker container."""
        result = await engine.execute("print('Hello from Docker')")
        assert result.success is True
        assert "Hello from Docker" in result.stdout
