"""
Evolution Node for the DeepAgent graph.

This node implements Story 3.1 and 3.2 safety mechanisms:
- Story 3.1: Syntax checking and dry run before code changes
- Story 3.2: Automatic rollback on test failures

The node:
1. Runs syntax check (ast.parse) on generated code
2. Runs dry-run import test
3. Creates .bak backup of target file
4. Applies the code patch
5. Runs pytest verification
6. Rolls back if pytest fails
"""
import core.logging
from typing import Dict, Any, Optional
from core.surgeon.safe_executor import check_syntax, dry_run_import, verify_code_before_commit
from core.version_control import FileBackupManager, run_pytest_verification, apply_patch_with_rollback


async def evolution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies code changes with comprehensive safety checks and rollback.
    
    Implements the full evolution pipeline:
    1. Gate 1: Syntax check (ast.parse)
    2. Gate 2: Dry run import
    3. Gate 3: Create .bak backup
    4. Gate 4: Apply patch
    5. Gate 5: Run pytest
    6. Rollback: Restore .bak if any gate fails
    
    Args:
        state: Dict containing:
            - code: str - The generated code to apply
            - filepath: str - Target file path
            - test_path: Optional[str] - Path to tests for verification
            
    Returns:
        Updated state with evolution results
    """
    code = state.get("code", "")
    filepath = state.get("filepath", "")
    test_path = state.get("test_path")
    
    result = {
        "evolution_success": False,
        "evolution_error": None,
        "gates_passed": [],
        "rolled_back": False
    }
    
    if not code or not filepath:
        result["evolution_error"] = "Missing code or filepath"
        core.logging.log_event("Evolution node: Missing required parameters", "ERROR")
        return result
    
    # --- GATE 1: Syntax Check ---
    core.logging.log_event(f"Evolution Gate 1: Checking syntax for {filepath}", "INFO")
    syntax_result = check_syntax(code)
    
    if not syntax_result["valid"]:
        result["evolution_error"] = (
            f"Syntax Error at line {syntax_result['line']}: {syntax_result['error']}"
        )
        core.logging.log_event(f"Evolution Gate 1 FAILED: {result['evolution_error']}", "WARNING")
        return result
    
    result["gates_passed"].append("syntax")
    core.logging.log_event("Evolution Gate 1 PASSED: Syntax OK", "INFO")
    
    # --- GATE 2: Dry Run Import ---
    core.logging.log_event(f"Evolution Gate 2: Dry run import test", "INFO")
    dry_result = dry_run_import(code)
    
    if not dry_result["success"]:
        result["evolution_error"] = (
            f"Import Error ({dry_result['error_type']}): {dry_result['error']}"
        )
        core.logging.log_event(f"Evolution Gate 2 FAILED: {result['evolution_error']}", "WARNING")
        return result
    
    result["gates_passed"].append("dry_import")
    core.logging.log_event("Evolution Gate 2 PASSED: Import OK", "INFO")
    
    # --- GATES 3-5: Apply with Rollback ---
    core.logging.log_event(f"Evolution Gates 3-5: Applying patch with rollback support", "INFO")
    
    patch_result = apply_patch_with_rollback(
        filepath=filepath,
        new_content=code,
        test_path=test_path
    )
    
    if patch_result["success"]:
        result["evolution_success"] = True
        result["gates_passed"].extend(["backup", "patch", "tests"])
        core.logging.log_event(
            f"Evolution complete for {filepath}: {patch_result['message']}",
            "INFO"
        )
    else:
        result["evolution_error"] = patch_result["message"]
        result["rolled_back"] = patch_result["rolled_back"]
        
        if patch_result["rolled_back"]:
            core.logging.log_event(
                f"Evolution rolled back for {filepath}: {patch_result['message']}",
                "WARNING"
            )
        else:
            core.logging.log_event(
                f"Evolution FAILED for {filepath}: {patch_result['message']}",
                "ERROR"
            )
    
    return result


def verify_evolution_safety(code: str, filepath: str) -> Dict[str, Any]:
    """
    Utility function to verify code is safe to evolve without applying it.
    
    Runs all verification gates but does not write to file.
    
    Args:
        code: Python code to verify
        filepath: Target path (for logging only)
        
    Returns:
        {
            "safe": bool,
            "gates_passed": list,
            "error": Optional[str]
        }
    """
    result = {
        "safe": False,
        "gates_passed": [],
        "error": None
    }
    
    # Gate 1: Syntax
    syntax_result = check_syntax(code)
    if not syntax_result["valid"]:
        result["error"] = f"Syntax Error: {syntax_result['error']}"
        return result
    result["gates_passed"].append("syntax")
    
    # Gate 2: Dry Import
    dry_result = dry_run_import(code)
    if not dry_result["success"]:
        result["error"] = f"Import Error: {dry_result['error']}"
        return result
    result["gates_passed"].append("dry_import")
    
    result["safe"] = True
    core.logging.log_event(
        f"Code verified safe for {filepath}: {result['gates_passed']}",
        "INFO"
    )
    
    return result
