import os
import subprocess
import core.logging

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