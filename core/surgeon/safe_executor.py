"""
Story 2.3: The "Surgeon" Sandbox - safe_execute_python tool

Provides isolated execution of Python code using Docker or subprocess fallback.
Code is tested in sandbox before being committed to the main codebase.
"""
import os
import sys
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional
from core.logging import log_event


# Check if Docker is available
def is_docker_available() -> bool:
    """Checks if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def safe_execute_python(
    code: str,
    timeout: int = 30,
    use_docker: bool = True,
    network_disabled: bool = True
) -> Dict[str, Any]:
    """
    Executes Python code in an isolated environment.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        use_docker: Whether to try Docker first (falls back to subprocess)
        network_disabled: Whether to disable network access in Docker
    
    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "execution_time": float,
            "sandbox_type": str  # "docker" or "subprocess"
        }
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
        "execution_time": 0.0,
        "sandbox_type": "unknown"
    }
    
    start_time = time.time()
    
    # Try Docker first if requested and available
    if use_docker and is_docker_available():
        result = _execute_in_docker(code, timeout, network_disabled)
        result["sandbox_type"] = "docker"
    else:
        # Fallback to subprocess with restrictions
        result = _execute_in_subprocess(code, timeout)
        result["sandbox_type"] = "subprocess"
    
    result["execution_time"] = time.time() - start_time
    log_event(f"Sandbox execution completed: {result['sandbox_type']}, exit_code={result['exit_code']}")
    
    return result


def _execute_in_docker(
    code: str,
    timeout: int,
    network_disabled: bool
) -> Dict[str, Any]:
    """Executes code in a Docker container."""
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1
    }
    
    try:
        from core.surgeon.sandbox import DockerSandbox
        
        sandbox = DockerSandbox()
        sandbox.ensure_image_exists()
        
        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Copy to container-relative path
            # The sandbox mounts the codebase, so we need to put the file there
            rel_path = os.path.basename(temp_path)
            
            # Execute in sandbox
            exit_code, stdout, stderr = sandbox.run_command(
                f"python3 /tmp/{rel_path}",
                timeout=timeout,
                network_disabled=network_disabled
            )
            
            result["exit_code"] = exit_code
            result["stdout"] = stdout
            result["stderr"] = stderr
            result["success"] = exit_code == 0
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        result["stderr"] = f"Docker execution failed: {str(e)}"
        log_event(f"Docker sandbox error: {e}")
    
    return result


def _execute_in_subprocess(code: str, timeout: int) -> Dict[str, Any]:
    """
    Executes code in a subprocess with restrictions.
    
    This is less isolated than Docker but provides basic sandboxing
    through subprocess timeout and environment isolation.
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1
    }
    
    # Write code to temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        # Create restricted environment
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",  # Don't inherit
            "HOME": tempfile.gettempdir(),  # Restricted home
            "LANG": "en_US.UTF-8",
        }
        
        # Run with timeout
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir()  # Run in temp dir
        )
        
        result["exit_code"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["success"] = proc.returncode == 0
        
    except subprocess.TimeoutExpired:
        result["stderr"] = f"Execution timed out after {timeout} seconds"
        result["exit_code"] = -1
        
    except Exception as e:
        result["stderr"] = f"Subprocess execution failed: {str(e)}"
        result["exit_code"] = -1
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return result


def verify_code_before_commit(
    code: str,
    filepath: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Verifies generated code in sandbox before writing to filesystem.
    
    Args:
        code: Python code to verify
        filepath: Target path for the code (for logging)
        timeout: Execution timeout
    
    Returns:
        {
            "verified": bool,
            "can_commit": bool,
            "execution_result": dict,
            "recommendation": str
        }
    """
    result = {
        "verified": False,
        "can_commit": False,
        "execution_result": None,
        "recommendation": ""
    }
    
    # Execute in sandbox
    exec_result = safe_execute_python(code, timeout=timeout)
    result["execution_result"] = exec_result
    
    if exec_result["success"]:
        result["verified"] = True
        result["can_commit"] = True
        result["recommendation"] = f"Code verified successfully. Safe to write to {filepath}"
    else:
        result["verified"] = True  # Verification ran, just failed
        result["can_commit"] = False
        
        if "SyntaxError" in exec_result["stderr"]:
            result["recommendation"] = "Code has syntax errors. Do not commit."
        elif "Timed out" in exec_result["stderr"]:
            result["recommendation"] = "Code execution timed out. May have infinite loop."
        else:
            result["recommendation"] = f"Code failed execution: {exec_result['stderr'][:200]}"
    
    log_event(f"Code verification for {filepath}: can_commit={result['can_commit']}")
    return result
