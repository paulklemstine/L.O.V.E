"""
Story 2.3: The "Surgeon" Sandbox - safe_execute_python tool
Story 3.1: The Surgeon's Sandbox - syntax check and dry run mode

Provides isolated execution of Python code using Docker or subprocess fallback.
Code is tested in sandbox before being committed to the main codebase.
"""
import ast
import os
import sys
import subprocess
import tempfile
import time
import importlib.util
from typing import Dict, Any, Optional
from core.logging import log_event


# --- Story 3.1: Syntax Check and Dry Run ---

def check_syntax(code: str) -> Dict[str, Any]:
    """
    Gate 1: Run ast.parse() on code string before any execution.
    
    Story 3.1: If ast.parse() fails, return the error to the LLM immediately.
    This prevents wasting execution cycles on syntactically invalid code.
    
    Args:
        code: Python code string to check
        
    Returns:
        {
            "valid": bool,
            "error": Optional[str],  # Error message
            "line": Optional[int],   # Line number of error
            "column": Optional[int], # Column offset of error
            "text": Optional[str]    # The problematic line text
        }
    """
    try:
        ast.parse(code)
        return {
            "valid": True,
            "error": None,
            "line": None,
            "column": None,
            "text": None
        }
    except SyntaxError as e:
        log_event(f"Syntax check failed at line {e.lineno}: {e.msg}", "WARNING")
        return {
            "valid": False,
            "error": str(e.msg),
            "line": e.lineno,
            "column": e.offset,
            "text": e.text.strip() if e.text else None
        }


def dry_run_import(code: str, module_name: str = "temp_module") -> Dict[str, Any]:
    """
    Gate 2: Saves code to temp file and attempts import to verify integrity.
    
    Story 3.1: "Dry Run" mode where code is saved to a temp file and imported
    to verify integrity before committing to the main codebase.
    
    This catches issues that ast.parse() doesn't find, such as:
    - Missing dependencies at import time
    - Circular imports
    - Invalid module-level expressions
    
    Args:
        code: Python code string to test
        module_name: Name for the temporary module
        
    Returns:
        {
            "success": bool,
            "error": Optional[str],
            "error_type": Optional[str],  # "ImportError", "AttributeError", etc.
            "module_loaded": bool
        }
    """
    result = {
        "success": False,
        "error": None,
        "error_type": None,
        "module_loaded": False
    }
    
    # First pass syntax check
    syntax_result = check_syntax(code)
    if not syntax_result["valid"]:
        result["error"] = f"Syntax error: {syntax_result['error']}"
        result["error_type"] = "SyntaxError"
        return result
    
    # Write to temp file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"{module_name}_",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name
        
        # Attempt to load the module using importlib
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        if spec is None or spec.loader is None:
            result["error"] = "Failed to create module spec"
            result["error_type"] = "ImportError"
            return result
        
        module = importlib.util.module_from_spec(spec)
        
        # Actually execute the module to catch runtime import errors
        try:
            spec.loader.exec_module(module)
            result["success"] = True
            result["module_loaded"] = True
            log_event(f"Dry run import successful for {module_name}", "INFO")
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            log_event(f"Dry run import failed: {type(e).__name__}: {e}", "WARNING")
            
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    return result


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
    network_disabled: bool = True,
    skip_syntax_check: bool = False  # Story 3.1: Option to skip if already checked
) -> Dict[str, Any]:
    """
    Executes Python code in an isolated environment.
    
    Story 3.1 Enhancement: Now runs ast.parse() syntax check FIRST.
    If syntax check fails, returns error immediately without execution.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        use_docker: Whether to try Docker first (falls back to subprocess)
        network_disabled: Whether to disable network access in Docker
        skip_syntax_check: Skip syntax check if already performed
    
    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "execution_time": float,
            "sandbox_type": str,  # "docker", "subprocess", or "syntax_check_failed"
            "syntax_error": Optional[dict]  # Details if syntax check failed
        }
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
        "execution_time": 0.0,
        "sandbox_type": "unknown",
        "syntax_error": None
    }
    
    start_time = time.time()
    
    # --- Story 3.1: GATE 1 - Syntax Check ---
    # Fail fast before any execution attempt
    if not skip_syntax_check:
        syntax_result = check_syntax(code)
        if not syntax_result["valid"]:
            result["success"] = False
            result["stderr"] = (
                f"Syntax Error at line {syntax_result['line']}, "
                f"column {syntax_result['column']}: {syntax_result['error']}\n"
                f"Problematic code: {syntax_result['text']}"
            )
            result["exit_code"] = 1
            result["sandbox_type"] = "syntax_check_failed"
            result["syntax_error"] = syntax_result
            result["execution_time"] = time.time() - start_time
            log_event(f"Syntax check gate blocked execution: {syntax_result['error']}", "WARNING")
            return result
    
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
    timeout: int = 30,
    run_dry_import: bool = True  # Story 3.1: Enable dry run import
) -> Dict[str, Any]:
    """
    Verifies generated code in sandbox before writing to filesystem.
    
    Story 3.1 Enhancement: Now includes:
    - Gate 1: ast.parse() syntax check
    - Gate 2: Dry run import (optional)
    - Gate 3: Full sandbox execution
    
    Args:
        code: Python code to verify
        filepath: Target path for the code (for logging)
        timeout: Execution timeout
        run_dry_import: Whether to run dry import verification (Gate 2)
    
    Returns:
        {
            "verified": bool,
            "can_commit": bool,
            "syntax_result": dict,   # Gate 1 results
            "dry_run_result": dict,  # Gate 2 results (if enabled)
            "execution_result": dict, # Gate 3 results
            "recommendation": str,
            "gates_passed": list[str]  # Which gates passed
        }
    """
    result = {
        "verified": False,
        "can_commit": False,
        "syntax_result": None,
        "dry_run_result": None,
        "execution_result": None,
        "recommendation": "",
        "gates_passed": []
    }
    
    # --- GATE 1: Syntax Check ---
    syntax_result = check_syntax(code)
    result["syntax_result"] = syntax_result
    
    if not syntax_result["valid"]:
        result["verified"] = True
        result["can_commit"] = False
        result["recommendation"] = (
            f"GATE 1 FAILED - Syntax Error at line {syntax_result['line']}: "
            f"{syntax_result['error']}. Do not commit."
        )
        log_event(f"Code verification for {filepath}: GATE 1 FAILED (syntax)", "WARNING")
        return result
    
    result["gates_passed"].append("syntax")
    
    # --- GATE 2: Dry Run Import (optional) ---
    if run_dry_import:
        dry_result = dry_run_import(code)
        result["dry_run_result"] = dry_result
        
        if not dry_result["success"]:
            result["verified"] = True
            result["can_commit"] = False
            result["recommendation"] = (
                f"GATE 2 FAILED - Import Error ({dry_result['error_type']}): "
                f"{dry_result['error']}. Do not commit."
            )
            log_event(f"Code verification for {filepath}: GATE 2 FAILED (import)", "WARNING")
            return result
        
        result["gates_passed"].append("dry_import")
    
    # --- GATE 3: Full Sandbox Execution ---
    exec_result = safe_execute_python(code, timeout=timeout, skip_syntax_check=True)
    result["execution_result"] = exec_result
    
    if exec_result["success"]:
        result["verified"] = True
        result["can_commit"] = True
        result["gates_passed"].append("execution")
        result["recommendation"] = (
            f"All gates passed ({', '.join(result['gates_passed'])}). "
            f"Safe to write to {filepath}"
        )
    else:
        result["verified"] = True
        result["can_commit"] = False
        
        if "Timed out" in exec_result["stderr"]:
            result["recommendation"] = "GATE 3 FAILED - Execution timed out. May have infinite loop."
        else:
            result["recommendation"] = f"GATE 3 FAILED - Execution error: {exec_result['stderr'][:200]}"
    
    log_event(f"Code verification for {filepath}: can_commit={result['can_commit']}, gates={result['gates_passed']}")
    return result


# =============================================================================
# Story 1.2: The "Do No Harm" Pre-Commit Check
# =============================================================================

class ForbiddenMutationError(Exception):
    """Raised when attempting to modify an immutable core file."""
    pass


def preflight_check(
    proposed_code: str,
    target_file: str = None,
    run_smoke_tests: bool = True,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Runs a comprehensive "dry run" before allowing code to be committed.
    
    Story 1.2: The "Do No Harm" Pre-Commit Check
    
    The check performs:
    1. Attempt to compile the code (compile(source, filename, 'exec'))
    2. Run tests/test_smoke.py against a temporary file (in sandbox)
    
    If the preflight check fails, returns a structured error that
    the Reasoning node can interpret.
    
    Args:
        proposed_code: The Python code to verify
        target_file: Optional path for context in error messages
        run_smoke_tests: Whether to run smoke tests (default True)
        timeout: Timeout for smoke test execution
        
    Returns:
        {
            "passed": bool,
            "compile_result": {
                "success": bool,
                "error": Optional[str],
                "line": Optional[int],
                "details": Optional[str]
            },
            "smoke_test_result": {
                "success": bool,
                "exit_code": int,
                "stdout": str,
                "stderr": str
            },
            "structured_error": Optional[dict]  # For Reasoning node
        }
    """
    result = {
        "passed": False,
        "compile_result": {
            "success": False,
            "error": None,
            "line": None,
            "details": None
        },
        "smoke_test_result": {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": ""
        },
        "structured_error": None
    }
    
    filename = target_file or "<proposed_code>"
    
    # ==========================================================================
    # GATE 1: Compile Check
    # ==========================================================================
    try:
        compile(proposed_code, filename, 'exec')
        result["compile_result"]["success"] = True
        log_event(f"Preflight compile check passed for {filename}", "INFO")
    except SyntaxError as e:
        result["compile_result"]["success"] = False
        result["compile_result"]["error"] = e.msg
        result["compile_result"]["line"] = e.lineno
        result["compile_result"]["details"] = e.text.strip() if e.text else None
        
        # Build structured error for Reasoning node
        result["structured_error"] = {
            "error_type": "SyntaxError",
            "message": f"Syntax error at line {e.lineno}: {e.msg}",
            "remediation": _suggest_syntax_remediation(e),
            "context": {
                "line": e.lineno,
                "column": e.offset,
                "code_snippet": e.text.strip() if e.text else ""
            }
        }
        
        log_event(f"Preflight compile check FAILED for {filename}: {e.msg}", "WARNING")
        return result
    
    # ==========================================================================
    # GATE 2: Smoke Test (if enabled)
    # ==========================================================================
    if run_smoke_tests:
        smoke_result = _run_smoke_tests_in_sandbox(proposed_code, target_file, timeout)
        result["smoke_test_result"] = smoke_result
        
        if not smoke_result["success"]:
            result["structured_error"] = {
                "error_type": "SmokeTestFailure",
                "message": f"Smoke tests failed with exit code {smoke_result['exit_code']}",
                "remediation": _suggest_smoke_test_remediation(smoke_result),
                "context": {
                    "exit_code": smoke_result["exit_code"],
                    "stderr_snippet": smoke_result["stderr"][:500] if smoke_result["stderr"] else "",
                    "stdout_snippet": smoke_result["stdout"][:500] if smoke_result["stdout"] else ""
                }
            }
            
            log_event(f"Preflight smoke test FAILED for {filename}", "WARNING")
            return result
    else:
        result["smoke_test_result"]["success"] = True
        result["smoke_test_result"]["exit_code"] = 0
    
    # All checks passed
    result["passed"] = True
    log_event(f"Preflight check PASSED for {filename}", "INFO")
    
    return result


def _suggest_syntax_remediation(error: SyntaxError) -> str:
    """Generates remediation suggestions based on the syntax error."""
    msg = error.msg.lower() if error.msg else ""
    
    if "unexpected eof" in msg:
        return "Check for missing closing brackets, parentheses, or quotes."
    elif "invalid syntax" in msg:
        return "Review the line for typos, missing operators, or incorrect indentation."
    elif "expected" in msg:
        return f"The parser expected something different. Check: {error.msg}"
    elif "indent" in msg:
        return "Check for inconsistent indentation (mixing tabs and spaces)."
    elif "unmatched" in msg or "never closed" in msg:
        return "Look for unclosed brackets, parentheses, or string quotes."
    else:
        return "Review the syntax around the indicated line for correctness."


def _suggest_smoke_test_remediation(smoke_result: Dict[str, Any]) -> str:
    """Generates remediation suggestions based on smoke test failure."""
    stderr = smoke_result.get("stderr", "").lower()
    
    if "importerror" in stderr or "modulenotfounderror" in stderr:
        return "A required module is missing. Check imports and dependencies."
    elif "attributeerror" in stderr:
        return "An object is missing an expected attribute. Check API compatibility."
    elif "typeerror" in stderr:
        return "A function was called with wrong argument types. Check function signatures."
    elif "nameerror" in stderr:
        return "A variable or function name is undefined. Check for typos or scope issues."
    elif "timeout" in stderr or "timed out" in stderr:
        return "The code may have an infinite loop or performance issue."
    elif smoke_result.get("exit_code") == -1:
        return "The test runner failed to execute. Check the sandbox environment."
    else:
        return "Review the test output for specific failure details."


def _run_smoke_tests_in_sandbox(
    proposed_code: str,
    target_file: str = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Runs smoke tests in the sandbox environment.
    
    Creates a temporary file with the proposed code and runs
    tests/test_smoke.py to verify the system still works.
    """
    result = {
        "success": False,
        "exit_code": -1,
        "stdout": "",
        "stderr": ""
    }
    
    temp_path = None
    backup_path = None
    
    try:
        # If we have a target file, we need to:
        # 1. Backup the original
        # 2. Write the proposed code
        # 3. Run smoke tests
        # 4. Restore the original
        
        if target_file and os.path.exists(target_file):
            backup_path = f"{target_file}.preflight_backup"
            
            # Backup original
            with open(target_file, "r", encoding="utf-8") as f:
                original_content = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            
            # Write proposed code
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(proposed_code)
            
            try:
                # Run smoke tests
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=_get_project_root()
                )
                
                result["exit_code"] = proc.returncode
                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
                result["success"] = proc.returncode == 0
                
            finally:
                # Restore original
                if backup_path and os.path.exists(backup_path):
                    with open(backup_path, "r", encoding="utf-8") as f:
                        original = f.read()
                    with open(target_file, "w", encoding="utf-8") as f:
                        f.write(original)
                    os.unlink(backup_path)
        else:
            # No target file - write to temp and run a basic syntax/import check
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(proposed_code)
                temp_path = f.name
            
            try:
                # Just try to import the temp file to check for runtime errors
                proc = subprocess.run(
                    [sys.executable, "-c", f"import importlib.util; spec = importlib.util.spec_from_file_location('test', '{temp_path}'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)"],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                result["exit_code"] = proc.returncode
                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
                result["success"] = proc.returncode == 0
                
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    except subprocess.TimeoutExpired:
        result["stderr"] = f"Smoke test timed out after {timeout} seconds"
        result["exit_code"] = -1
        
    except Exception as e:
        result["stderr"] = f"Smoke test execution error: {str(e)}"
        result["exit_code"] = -1
    
    return result


def _get_project_root() -> str:
    """Returns the project root directory."""
    # Navigate up from core/surgeon/safe_executor.py to get project root
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current))

