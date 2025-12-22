
import subprocess
import logging
import os
from typing import Tuple

class GitHandler:
    def __init__(self, repo_path: str = "."):
        self.repo_path = os.path.abspath(repo_path)

    def _run_git(self, args: list) -> Tuple[int, str, str]:
        """Runs a git command in the repo directory."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            logging.error(f"Failed to run git command {args}: {e}")
            return -1, "", str(e)

    def branch_exists(self, branch_name: str) -> bool:
        """Checks if a branch exists locally."""
        code, stdout, _ = self._run_git(["branch", "--list", branch_name])
        return branch_name in stdout

    def create_branch(self, branch_name: str) -> bool:
        """
        Creates a new branch. If it exists, does nothing (returns True).
        """
        if self.branch_exists(branch_name):
            logging.info(f"Branch {branch_name} already exists.")
            return True
            
        logging.info(f"Creating branch {branch_name}...")
        code, _, stderr = self._run_git(["branch", branch_name])
        if code != 0:
            logging.error(f"Failed to create branch {branch_name}: {stderr}")
            return False
        return True

    def checkout_branch(self, branch_name: str) -> bool:
        """
        Checks out the specified branch.
        """
        logging.info(f"Checking out branch {branch_name}...")
        code, _, stderr = self._run_git(["checkout", branch_name])
        if code != 0:
            # Try creating and checking out if it's new? 
            # Story says "Agent creates/checkouts".
            # If checkout fails, assume it might not exist or verify existence first.
            # Usually checkout -b is used if new.
            # But let's assume create_branch was called or we strictly follow steps.
            logging.error(f"Failed to checkout branch {branch_name}: {stderr}")
            return False
        return True

    def commit_changes(self, message: str) -> bool:
        """
        Stages all changes and commits them.
        """
        # Stage
        code, _, stderr = self._run_git(["add", "."])
        if code != 0:
            logging.error(f"Failed to stage changes: {stderr}")
            return False
            
        # Commit
        code, _, stderr = self._run_git(["commit", "-m", message])
        if code != 0:
            # Commit returns 1 if nothing to commit, which is not necessarily a failure of "logic"
            # but usually we expect something.
            status_code, status_out, _ = self._run_git(["status"])
            if "nothing to commit" in status_out:
                 logging.info("Nothing to commit.")
                 return True
            logging.error(f"Failed to commit changes: {stderr}")
            return False
        return True

    def create_pr(self, title: str, body: str, base: str = "main") -> bool:
        """
        Simulates creating a PR. In real scenario, uses 'gh' CLI.
        """
        # Check if gh is installed or just log
        code, _, _ = self._run_git(["help", "gh"]) # 'git help gh' ? No, 'gh' is separate.
        # Just logging for this task
        logging.info(f"SIMULATION: Creating PR to merge into {base}")
        logging.info(f"Title: {title}")
        logging.info(f"Body: {body}")
        
        # We assume success for simulation
        return True
