import os
import shutil
import subprocess
import logging
from rich.console import Console

class Sandbox:
    """
    Manages a temporary, isolated environment for testing code changes.
    """
    def __init__(self, repo_url, base_dir="love_sandbox"):
        self.repo_url = repo_url
        self.base_dir = os.path.abspath(base_dir)
        self.sandbox_path = None
        self.console = Console()
        os.makedirs(self.base_dir, exist_ok=True)

    def _run_command(self, command, cwd):
        """Executes a shell command in a specified directory and logs the output."""
        logging.info(f"Running command in sandbox: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600  # 10-minute timeout for tests
            )
            if process.stdout:
                logging.info(f"Sandbox stdout:\n{process.stdout}")
            if process.stderr:
                logging.warning(f"Sandbox stderr:\n{process.stderr}")
            return process.returncode, process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            logging.error("Sandbox command timed out.")
            return -1, "", "Command timed out after 10 minutes."
        except Exception as e:
            logging.error(f"Sandbox command failed: {e}")
            return -1, "", str(e)

    def create(self, branch_name):
        """Creates a new sandbox by cloning a specific branch of the repository."""
        self.sandbox_path = os.path.join(self.base_dir, branch_name)
        if os.path.exists(self.sandbox_path):
            self.console.print(f"[yellow]Cleaning up existing sandbox at: {self.sandbox_path}[/yellow]")
            shutil.rmtree(self.sandbox_path)

        self.console.print(f"[cyan]Creating new sandbox for branch '{branch_name}'...[/cyan]")

        # Clone the specific branch into the sandbox directory
        clone_cmd = ["git", "clone", "--branch", branch_name, "--single-branch", self.repo_url, self.sandbox_path]
        returncode, stdout, stderr = self._run_command(clone_cmd, cwd=self.base_dir)

        if returncode != 0:
            self.console.print(f"[bold red]Failed to create sandbox for branch '{branch_name}'.[/bold red]")
            logging.error(f"Git clone failed for branch {branch_name}: {stderr}")
            self.sandbox_path = None
            return False

        self.console.print(f"[green]Sandbox created successfully at: {self.sandbox_path}[/green]")
        return True

    def run_tests(self):
        """
        Runs the pytest test suite within the sandbox.
        Assumes pytest is installed in the environment.
        """
        if not self.sandbox_path:
            self.console.print("[bold red]Cannot run tests, sandbox is not created.[/bold red]")
            return False, "Sandbox not created."

        self.console.print("[cyan]Running pytest suite in sandbox...[/cyan]")

        # We can add more setup steps here, like installing dependencies from a requirements.txt if needed.
        # For now, we assume the environment is ready.

        test_cmd = ["pytest"]
        returncode, stdout, stderr = self._run_command(test_cmd, cwd=self.sandbox_path)

        if returncode == 0:
            self.console.print("[bold green]All sandbox tests passed.[/bold green]")
            return True, stdout
        else:
            self.console.print("[bold red]Sandbox tests failed.[/bold red]")
            # Combine stdout and stderr for a complete failure log
            failure_log = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            return False, failure_log

    def destroy(self):
        """Removes the sandbox directory."""
        if self.sandbox_path and os.path.exists(self.sandbox_path):
            self.console.print(f"[cyan]Destroying sandbox: {self.sandbox_path}[/cyan]")
            try:
                shutil.rmtree(self.sandbox_path)
                self.sandbox_path = None
                logging.info("Sandbox destroyed successfully.")
            except Exception as e:
                logging.error(f"Failed to destroy sandbox: {e}")
                self.console.print(f"[bold red]Error destroying sandbox: {e}[/bold red]")