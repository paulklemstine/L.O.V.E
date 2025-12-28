import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.logging import log_event


# Constants
EVOLUTION_LOG_FILE = "EVOLUTION_LOG.md"
DEFAULT_CRASH_THRESHOLD = 3


class SystemIntegrityMonitor:
    """
    Monitors system health, detects crash loops, and performs auto-rollback.
    Story 2.1: The Immune System
    """
    def __init__(self, crash_threshold: int = DEFAULT_CRASH_THRESHOLD):
        self.component_states = {}
        self.crash_threshold = crash_threshold
        self.exit_code_history: List[int] = []
    
    def record_exit_code(self, exit_code: int) -> None:
        """Records an exit code for crash loop detection."""
        self.exit_code_history.append(exit_code)
        # Keep only last N*2 entries
        max_history = self.crash_threshold * 2
        if len(self.exit_code_history) > max_history:
            self.exit_code_history = self.exit_code_history[-max_history:]
    
    def detect_crash_loop(self, exit_codes: List[int] = None) -> bool:
        """
        Detects if recent exit codes indicate a crash loop.
        
        Args:
            exit_codes: Optional list to check. If None, uses internal history.
            
        Returns:
            True if crash loop detected (N consecutive non-zero exits)
        """
        codes = exit_codes if exit_codes is not None else self.exit_code_history
        
        if len(codes) < self.crash_threshold:
            return False
        
        # Check last N exit codes
        recent = codes[-self.crash_threshold:]
        return all(code != 0 for code in recent)
    
    def detect_exception_loop(self, exceptions: List[str]) -> bool:
        """
        Detects if the same exception is occurring repeatedly.
        
        Args:
            exceptions: List of recent exception messages
            
        Returns:
            True if same exception occurred threshold times
        """
        if len(exceptions) < self.crash_threshold:
            return False
        
        recent = exceptions[-self.crash_threshold:]
        # Check if all recent exceptions are the same
        return len(set(recent)) == 1
    
    def get_current_commit(self) -> Optional[str]:
        """Gets the current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            log_event(f"Failed to get current commit: {e}")
        return None
    
    def create_backup_branch(self) -> Optional[str]:
        """Creates a backup branch before rollback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"backup_before_rollback_{timestamp}"
        
        try:
            result = subprocess.run(
                ["git", "branch", branch_name],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                log_event(f"Created backup branch: {branch_name}")
                return branch_name
        except Exception as e:
            log_event(f"Failed to create backup branch: {e}")
        return None
    
    def auto_rollback(self, reason: str, traceback_str: str = "") -> Dict[str, Any]:
        """
        Executes git revert HEAD and logs to EVOLUTION_LOG.md.
        
        Args:
            reason: Description of why rollback is needed
            traceback_str: Optional traceback for logging
            
        Returns:
            Dict with rollback result: {success, commit_reverted, backup_branch, message}
        """
        result = {
            "success": False,
            "commit_reverted": None,
            "backup_branch": None,
            "message": ""
        }
        
        # Get current commit before revert
        current_commit = self.get_current_commit()
        if not current_commit:
            result["message"] = "Failed to get current commit"
            return result
        
        result["commit_reverted"] = current_commit
        
        # Create backup branch
        backup_branch = self.create_backup_branch()
        result["backup_branch"] = backup_branch
        
        # Execute git revert
        try:
            revert_result = subprocess.run(
                ["git", "revert", "--no-commit", "HEAD"],
                capture_output=True, text=True, timeout=30
            )
            
            if revert_result.returncode != 0:
                # Revert failed (possibly due to conflicts)
                result["message"] = f"Git revert failed: {revert_result.stderr}"
                # Abort the revert
                subprocess.run(["git", "revert", "--abort"], capture_output=True, timeout=10)
                self.log_failure_to_evolution_log(current_commit, reason, traceback_str, success=False)
                return result
            
            # Commit the revert
            commit_result = subprocess.run(
                ["git", "commit", "-m", f"Auto-rollback: {reason[:50]}"],
                capture_output=True, text=True, timeout=30
            )
            
            if commit_result.returncode == 0:
                result["success"] = True
                result["message"] = f"Successfully reverted commit {current_commit[:8]}"
                log_event(f"Auto-rollback successful: reverted {current_commit[:8]}")
            else:
                result["message"] = f"Commit failed: {commit_result.stderr}"
            
        except subprocess.TimeoutExpired:
            result["message"] = "Git revert timed out"
        except Exception as e:
            result["message"] = f"Rollback error: {str(e)}"
        
        # Log to EVOLUTION_LOG.md
        self.log_failure_to_evolution_log(current_commit, reason, traceback_str, result["success"])
        
        return result
    
    def log_failure_to_evolution_log(
        self, 
        commit_hash: str, 
        reason: str, 
        traceback_str: str = "",
        success: bool = False
    ) -> None:
        """
        Creates an entry in EVOLUTION_LOG.md detailing the failure.
        
        Args:
            commit_hash: The commit that caused the failure
            reason: Description of the failure
            traceback_str: Optional traceback
            success: Whether the rollback was successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "✅ Reverted" if success else "❌ Revert Failed"
        
        # Truncate reason and traceback for table
        reason_short = reason[:80].replace("|", "\\|").replace("\n", " ")
        
        entry = f"| {timestamp} | {commit_hash[:8]} | {status} | {reason_short} |\n"
        
        try:
            # Append to EVOLUTION_LOG.md
            with open(EVOLUTION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            
            log_event(f"Logged failure to {EVOLUTION_LOG_FILE}: {commit_hash[:8]}")
            
        except Exception as e:
            log_event(f"Failed to write to {EVOLUTION_LOG_FILE}: {e}")

    def evaluate_component_status(self, component_data):
        """
        Evaluates the status of a system component based on provided data.
        """
        evaluation_report = {
            "component": component_data.get("name", "Unknown"),
            "status": "nominal",
            "discrepancies": [],
        }

        # Placeholder for talent_scout evaluation logic
        if component_data.get("name") == "talent_scout":
            if component_data.get("profiles_found", 0) == 0:
                evaluation_report["status"] = "suboptimal"
                evaluation_report["discrepancies"].append("No profiles found.")

        # Placeholder for research_and_evolve evaluation logic
        if component_data.get("name") == "research_and_evolve":
            if not component_data.get("user_stories_generated", False):
                evaluation_report["status"] = "inefficient"
                evaluation_report["discrepancies"].append("No user stories were generated.")

        return evaluation_report

    def suggest_enhancements(self, evaluation_report):
        """
        Suggests enhancements based on an evaluation report.
        """
        suggestions = []

        if evaluation_report.get("status") == "suboptimal":
            suggestions.append("Consider broadening the search keywords or platforms.")

        if evaluation_report.get("status") == "inefficient":
            suggestions.append("Review the research data sources and analysis algorithms.")

        return suggestions

    def track_evolution(self, component_name, current_state):
        """
        Tracks the evolution of a system component.
        """
        previous_state = self.component_states.get(component_name, {})
        evolution_report = {
            "previous_state": previous_state,
            "current_state": current_state,
            "changes": [],
        }

        # Placeholder for evolution tracking logic
        if previous_state.get("profiles_found", 0) < current_state.get("profiles_found", 0):
            evolution_report["changes"].append("Increased number of profiles found.")

        if not previous_state.get("user_stories_generated", False) and current_state.get("user_stories_generated", False):
            evolution_report["changes"].append("User stories were generated for the first time.")

        self.component_states[component_name] = current_state
        return evolution_report


# Global instance for easy access
system_monitor = SystemIntegrityMonitor()

