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

import re
from datetime import datetime


class GitManager:
    """
    Manages Git operations for L.O.V.E. evolution workflows.
    
    Story 3.1: Provides evolution-aware branch naming and automated PR generation.
    """
    
    def __init__(self, repo_path=".", activity_log_path: str = None):
        self.repo_path = repo_path
        self.token = os.environ.get("GITHUB_TOKEN")
        self.activity_log_path = activity_log_path or os.path.join(repo_path, "activity_log.md")

    def _run_cmd(self, cmd_list, check: bool = True) -> Optional[str]:
        try:
            result = subprocess.run(
                cmd_list, cwd=self.repo_path, check=check, 
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            core.logging.log_event(f"Git Command Failed: {e.stderr}", "ERROR")
            return None

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_-]+', '-', text)
        return text[:50]  # Limit length

    def _log_activity(self, action: str, details: str) -> None:
        """Append to activity log for traceability."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"| {timestamp} | {action} | {details} |\n"
            
            # Ensure file exists with header
            if not os.path.exists(self.activity_log_path):
                with open(self.activity_log_path, 'w') as f:
                    f.write("# Activity Log\n\n| Timestamp | Action | Details |\n|---|---|---|\n")
            
            with open(self.activity_log_path, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            core.logging.log_event(f"Failed to log activity: {e}", "WARNING")

    # --- Story 3.1: Evolution-Aware Branch Management ---
    
    def create_evolution_branch(self, intent_slug: str) -> Optional[str]:
        """
        Creates a branch with evolution naming convention.
        
        Story 3.1: Branch name format: feature/evolution-{timestamp}-{intent_slug}
        
        Args:
            intent_slug: Human-readable description of the evolution intent
            
        Returns:
            Branch name if created, None on failure
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        slug = self._slugify(intent_slug)
        branch_name = f"feature/evolution-{timestamp}-{slug}"
        
        # Ensure we're on main first
        self._run_cmd(["git", "checkout", "main"], check=False)
        self._run_cmd(["git", "pull", "origin", "main"], check=False)
        
        result = self._run_cmd(["git", "checkout", "-b", branch_name])
        if result is not None:
            self._log_activity("BRANCH_CREATED", branch_name)
            core.logging.log_event(f"Created evolution branch: {branch_name}", "INFO")
            return branch_name
        return None

    def get_current_branch(self) -> Optional[str]:
        """Returns the current branch name."""
        return self._run_cmd(["git", "branch", "--show-current"])

    def commit_evolution(self, files: List[str], message: str, intent: str = None) -> bool:
        """
        Stages and commits multiple files with semantic message.
        
        Story 3.1: Commits all evolution-related changes atomically.
        
        Args:
            files: List of file paths to stage
            message: Commit message
            intent: Optional intent description for activity log
            
        Returns:
            True if commit succeeded
        """
        # Stage all files
        for file_path in files:
            result = self._run_cmd(["git", "add", file_path])
            if result is None:
                core.logging.log_event(f"Failed to stage: {file_path}", "WARNING")
        
        # Commit
        result = self._run_cmd(["git", "commit", "-m", message])
        if result is not None:
            self._log_activity("COMMIT", f"{message[:80]}... ({len(files)} files)")
            return True
        return False

    def create_evolution_pr(
        self, 
        branch_name: str, 
        title: str,
        description: str,
        base_branch: str = "main"
    ) -> Optional[str]:
        """
        Creates a Pull Request for an evolution branch.
        
        Story 3.1: Generates PR with semantic description and logs the URL.
        
        Args:
            branch_name: Head branch name
            title: PR title
            description: PR body (markdown)
            base_branch: Target branch (default: main)
            
        Returns:
            PR URL if created, None on failure
        """
        # Push branch first
        push_result = self._run_cmd(["git", "push", "-u", "origin", branch_name])
        if push_result is None:
            core.logging.log_event(f"Failed to push branch: {branch_name}", "ERROR")
            return None
        
        if not self.token:
            core.logging.log_event("No GITHUB_TOKEN found. Cannot create PR.", "WARNING")
            # Still log the branch for manual PR creation
            self._log_activity("PR_READY", f"Branch pushed: {branch_name} (manual PR required)")
            return None

        env = os.environ.copy()
        env["GITHUB_TOKEN"] = self.token
        
        try:
            cmd = [
                "gh", "pr", "create", 
                "--title", title, 
                "--body", description, 
                "--head", branch_name, 
                "--base", base_branch
            ]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                pr_url = result.stdout.strip()
                self._log_activity("PR_CREATED", pr_url)
                core.logging.log_event(f"Created PR: {pr_url}", "INFO")
                return pr_url
            else:
                core.logging.log_event(f"GH CLI failed: {result.stderr}", "ERROR")
                self._log_activity("PR_FAILED", result.stderr[:100])
                return None
                
        except FileNotFoundError:
            core.logging.log_event("GitHub CLI 'gh' not installed.", "ERROR")
            self._log_activity("PR_READY", f"Branch {branch_name} ready (gh CLI not available)")
            return None

    # --- Legacy Methods (preserved for compatibility) ---

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
        return self._run_cmd(["git", "push", "-u", "origin", branch_name])

    def create_pull_request(self, title: str, body: str, head_branch: str, base_branch="main"):
        """Uses GitHub CLI (gh) or API to create PR."""
        return self.create_evolution_pr(head_branch, title, body, base_branch)