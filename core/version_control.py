import os
import subprocess
import shutil
import time
from typing import Dict, Any, Optional, List
import core.logging


# --- Story 3.2: File Backup Manager ---

class FileBackupManager:
    """
    Manages .bak file creation and rollback for safe code evolution.
    
    Story 3.2: Creates a .bak of any file before editing, enabling
    automatic rollback if the system fails to start or tests fail.
    """
    
    def __init__(self, backup_dir: str = None):
        """
        Initialize the backup manager.
        
        Args:
            backup_dir: Optional directory for backups. If None, uses same dir as original.
        """
        self.backup_dir = backup_dir
        self.backup_registry: Dict[str, str] = {}  # Maps original -> backup path
    
    def create_backup(self, filepath: str) -> Optional[str]:
        """
        Creates a .bak of a file before editing.
        
        Args:
            filepath: Path to the file to backup
            
        Returns:
            Path to the backup file, or None if backup failed
        """
        if not os.path.exists(filepath):
            core.logging.log_event(f"Cannot backup non-existent file: {filepath}", "WARNING")
            return None
        
        try:
            # Generate backup path
            if self.backup_dir:
                os.makedirs(self.backup_dir, exist_ok=True)
                filename = os.path.basename(filepath)
                timestamp = int(time.time())
                backup_path = os.path.join(self.backup_dir, f"{filename}.{timestamp}.bak")
            else:
                backup_path = f"{filepath}.bak"
            
            # Remove existing backup if present
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            # Copy file to backup
            shutil.copy2(filepath, backup_path)
            
            # Register the backup
            self.backup_registry[filepath] = backup_path
            
            core.logging.log_event(f"Created backup: {filepath} -> {backup_path}", "INFO")
            return backup_path
            
        except Exception as e:
            core.logging.log_event(f"Failed to create backup for {filepath}: {e}", "ERROR")
            return None
    
    def restore_backup(self, filepath: str) -> bool:
        """
        Restores a file from its .bak backup.
        
        Args:
            filepath: Path to the original file to restore
            
        Returns:
            True if restore succeeded, False otherwise
        """
        # Find backup path
        backup_path = self.backup_registry.get(filepath)
        
        if not backup_path:
            # Try default .bak location
            backup_path = f"{filepath}.bak"
        
        if not os.path.exists(backup_path):
            core.logging.log_event(f"No backup found for {filepath}", "WARNING")
            return False
        
        try:
            # Restore from backup
            shutil.copy2(backup_path, filepath)
            core.logging.log_event(f"Restored {filepath} from backup", "INFO")
            return True
            
        except Exception as e:
            core.logging.log_event(f"Failed to restore {filepath}: {e}", "ERROR")
            return False
    
    def cleanup_backup(self, filepath: str) -> bool:
        """
        Removes .bak file after successful commit.
        
        Args:
            filepath: Path to the original file
            
        Returns:
            True if cleanup succeeded, False otherwise
        """
        backup_path = self.backup_registry.get(filepath, f"{filepath}.bak")
        
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
                if filepath in self.backup_registry:
                    del self.backup_registry[filepath]
                core.logging.log_event(f"Cleaned up backup: {backup_path}", "DEBUG")
                return True
            except Exception as e:
                core.logging.log_event(f"Failed to cleanup backup {backup_path}: {e}", "WARNING")
                return False
        
        return True  # No backup to clean up
    
    def get_backup_path(self, filepath: str) -> Optional[str]:
        """Returns the backup path for a file if it exists."""
        backup_path = self.backup_registry.get(filepath, f"{filepath}.bak")
        return backup_path if os.path.exists(backup_path) else None


def run_pytest_verification(
    test_path: str = "tests/",
    timeout: int = 120,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Runs pytest and returns results.
    
    Story 3.2: The EvolutionNode must run pytest after applying a patch.
    If pytest fails critically (exit code != 0), trigger rollback.
    
    Args:
        test_path: Path to tests directory or specific test file
        timeout: Maximum time for test execution
        verbose: Whether to include full output
        
    Returns:
        {
            "success": bool,
            "exit_code": int,
            "passed": int,
            "failed": int,
            "errors": int,
            "output": str,
            "should_rollback": bool  # True if exit_code != 0
        }
    """
    result = {
        "success": False,
        "exit_code": -1,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "output": "",
        "should_rollback": False
    }
    
    try:
        # Run pytest with parseable output format
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "-v" if verbose else "-q",
            "--tb=short"  # Short traceback format
        ]
        
        core.logging.log_event(f"Running pytest: {' '.join(cmd)}", "INFO")
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        result["exit_code"] = proc.returncode
        result["output"] = proc.stdout + proc.stderr
        result["success"] = proc.returncode == 0
        result["should_rollback"] = proc.returncode != 0
        
        # Parse output for pass/fail counts
        output_lower = result["output"].lower()
        
        # Look for pytest summary line like "2 passed, 1 failed"
        import re
        passed_match = re.search(r'(\d+)\s+passed', output_lower)
        failed_match = re.search(r'(\d+)\s+failed', output_lower)
        error_match = re.search(r'(\d+)\s+error', output_lower)
        
        if passed_match:
            result["passed"] = int(passed_match.group(1))
        if failed_match:
            result["failed"] = int(failed_match.group(1))
        if error_match:
            result["errors"] = int(error_match.group(1))
        
        log_msg = f"Pytest: {result['passed']} passed, {result['failed']} failed, {result['errors']} errors"
        log_level = "INFO" if result["success"] else "WARNING"
        core.logging.log_event(log_msg, log_level)
        
    except subprocess.TimeoutExpired:
        result["output"] = f"Pytest timed out after {timeout} seconds"
        result["should_rollback"] = True
        core.logging.log_event(result["output"], "ERROR")
        
    except Exception as e:
        result["output"] = f"Pytest execution failed: {e}"
        result["should_rollback"] = True
        core.logging.log_event(result["output"], "ERROR")
    
    return result


def apply_patch_with_rollback(
    filepath: str,
    new_content: str,
    test_path: str = None,
    backup_manager: FileBackupManager = None
) -> Dict[str, Any]:
    """
    Applies a patch with automatic rollback on failure.
    
    Story 3.2 Implementation:
    1. Creates .bak of original file
    2. Writes new content
    3. Runs pytest (if test_path provided)
    4. If pytest fails: restores .bak and logs failure
    5. If pytest passes: removes .bak
    
    Args:
        filepath: Path to the file to modify
        new_content: New content to write
        test_path: Optional path to tests to run for verification
        backup_manager: Optional FileBackupManager instance
        
    Returns:
        {
            "success": bool,
            "rolled_back": bool,
            "backup_path": str,
            "test_result": Optional[dict],
            "message": str
        }
    """
    result = {
        "success": False,
        "rolled_back": False,
        "backup_path": None,
        "test_result": None,
        "message": ""
    }
    
    # Use provided backup manager or create new one
    if backup_manager is None:
        backup_manager = FileBackupManager()
    
    # Step 1: Create backup
    if os.path.exists(filepath):
        backup_path = backup_manager.create_backup(filepath)
        result["backup_path"] = backup_path
        
        if not backup_path:
            result["message"] = "Failed to create backup. Aborting patch."
            core.logging.log_event(result["message"], "ERROR")
            return result
    
    # Step 2: Write new content
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        core.logging.log_event(f"Wrote patch to {filepath}", "INFO")
        
    except Exception as e:
        result["message"] = f"Failed to write patch: {e}"
        core.logging.log_event(result["message"], "ERROR")
        
        # Attempt rollback
        if result["backup_path"]:
            if backup_manager.restore_backup(filepath):
                result["rolled_back"] = True
                result["message"] += " Rolled back successfully."
        
        return result
    
    # Step 3: Run tests if test_path provided
    if test_path:
        test_result = run_pytest_verification(test_path)
        result["test_result"] = test_result
        
        # Step 4: Rollback if tests failed
        if test_result["should_rollback"]:
            core.logging.log_event(
                f"Tests failed after patch. Initiating rollback for {filepath}",
                "WARNING"
            )
            
            if backup_manager.restore_backup(filepath):
                result["rolled_back"] = True
                result["message"] = (
                    f"Patch rolled back due to test failures. "
                    f"Passed: {test_result['passed']}, Failed: {test_result['failed']}"
                )
            else:
                result["message"] = "CRITICAL: Tests failed and rollback failed!"
            
            core.logging.log_event(result["message"], "ERROR" if not result["rolled_back"] else "WARNING")
            return result
        
        # Step 5: Tests passed, cleanup backup
        backup_manager.cleanup_backup(filepath)
        result["success"] = True
        result["message"] = (
            f"Patch applied and verified. "
            f"Tests: {test_result['passed']} passed, {test_result['failed']} failed"
        )
        
    else:
        # No tests specified, consider success
        result["success"] = True
        result["message"] = f"Patch applied to {filepath}. No tests were run."
        # Keep backup for manual verification
    
    core.logging.log_event(result["message"], "INFO")
    return result


# --- Git Operations ---

class GitManager:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.token = os.environ.get("GITHUB_TOKEN")

    def _run_cmd(self, cmd_list):
        try:
            result = subprocess.run(
                cmd_list, cwd=self.repo_path, check=True, 
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            core.logging.log_event(f"Git Command Failed: {e.stderr}", "ERROR")
            return None

    def create_branch(self, branch_name: str):
        """Creates and switches to a new branch."""
        current = self._run_cmd(["git", "branch", "--show-current"])
        if current == branch_name:
            return True
        return self._run_cmd(["git", "checkout", "-b", branch_name])

    def commit_changes(self, file_to_add: str, commit_message: str):
        """Stages and commits a file."""
        self._run_cmd(["git", "add", file_to_add])
        return self._run_cmd(["git", "commit", "-m", commit_message])

    def push_changes(self, branch_name: str):
        """Pushes branch to remote."""
        # Use token in URL for auth if needed, or rely on system auth
        return self._run_cmd(["git", "push", "-u", "origin", branch_name])

    def create_pull_request(self, title: str, body: str, head_branch: str, base_branch="main"):
        """Uses GitHub CLI (gh) or API to create PR."""
        if not self.token:
            core.logging.log_event("No GITHUB_TOKEN found. Cannot create PR.", "WARNING")
            return False

        # Try using 'gh' CLI if installed
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = self.token
        
        try:
            cmd = [
                "gh", "pr", "create", 
                "--title", title, 
                "--body", body, 
                "--head", head_branch, 
                "--base", base_branch
            ]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                core.logging.log_event(f"GH CLI failed: {result.stderr}", "ERROR")
                return None
        except FileNotFoundError:
            core.logging.log_event("GitHub CLI 'gh' not installed.", "ERROR")
            return None