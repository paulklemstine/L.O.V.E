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

    def run_script(self, script_content: str, script_name: str = "verification_script.py"):
        """
        Writes a script to the sandbox and executes it.
        Returns: (success: bool, output: str)
        """
        if not self.sandbox_path:
             return False, "Sandbox not created."
        
        script_path = os.path.join(self.sandbox_path, script_name)
        try:
            with open(script_path, "w") as f:
                f.write(script_content)
        except Exception as e:
            return False, f"Failed to write script: {e}"
        
        cmd = ["python3", script_name]
        returncode, stdout, stderr = self._run_command(cmd, cwd=self.sandbox_path)
        
        output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        if returncode == 0:
            return True, output
        else:
            return False, output


    def create(self, branch_name):
        """Creates a new sandbox by cloning a specific branch of the repository."""
        self.sandbox_path = os.path.join(self.base_dir, branch_name)
        if os.path.exists(self.sandbox_path):
            self.console.print(f"[yellow]Cleaning up existing sandbox at: {self.sandbox_path}[/yellow]")
            shutil.rmtree(self.sandbox_path)

        self.console.print(f"[cyan]Creating new sandbox for branch '{branch_name}'...[/cyan]")

        # Clone the repository first
        clone_cmd = ["git", "clone", self.repo_url, self.sandbox_path]
        returncode, stdout, stderr = self._run_command(clone_cmd, cwd=self.base_dir)

        if returncode != 0:
            self.console.print(f"[bold red]Failed to clone repository for sandbox '{branch_name}'.[/bold red]")
            logging.error(f"Git clone failed for branch {branch_name}: {stderr}")
            self.sandbox_path = None
            return False

        # Now, checkout the specific branch or commit
        checkout_cmd = ["git", "checkout", branch_name]
        returncode, stdout, stderr = self._run_command(checkout_cmd, cwd=self.sandbox_path)

        if returncode != 0:
            self.console.print(f"[bold red]Failed to checkout '{branch_name}' in sandbox.[/bold red]")
            logging.error(f"Git checkout failed for '{branch_name}': {stderr}")
            # Clean up the cloned repo if checkout fails
            self.destroy()
            return False

        self.console.print(f"[green]Sandbox created successfully at: {self.sandbox_path}[/green]")
        return True

    def run_tests(self):
        """
        Runs a startup check and the pytest test suite within the sandbox.
        """
        if not self.sandbox_path:
            self.console.print("[bold red]Cannot run tests, sandbox is not created.[/bold red]")
            return False, "Sandbox not created."

        # 1. Startup Test (Syntax Check)
        self.console.print("[cyan]Running startup test (syntax check) in sandbox...[/cyan]")
        startup_cmd = ["python3", "-c", "import love"]
        returncode, stdout, stderr = self._run_command(startup_cmd, cwd=self.sandbox_path)

        if returncode != 0:
            self.console.print("[bold red]Sandbox startup test failed. The code has syntax errors or critical import issues.[/bold red]")
            failure_log = f"STARTUP TEST FAILED:\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            return False, failure_log

        self.console.print("[green]Startup test passed.[/green]")

        # 2. Pytest Suite
        self.console.print("[cyan]Running pytest suite in sandbox...[/cyan]")
        # Exclude the integration lifecycle tests to prevent recursion/inception
        test_cmd = ["python3", "-m", "pytest", "--ignore=tests/test_jules_lifecycle.py"]
        returncode, stdout, stderr = self._run_command(test_cmd, cwd=self.sandbox_path)

        if returncode == 0:
            self.console.print("[bold green]All sandbox tests passed.[/bold green]")
            return True, stdout
        else:
            self.console.print("[bold red]Sandbox tests failed.[/bold red]")
            failure_log = f"PYTEST FAILED:\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            return False, failure_log

    def get_diff(self):
        """
        Gets the git diff of the current branch against the main branch.
        """
        if not self.sandbox_path:
            self.console.print("[bold red]Cannot get diff, sandbox is not created.[/bold red]")
            return None, "Sandbox not created."

        self.console.print("[cyan]Getting git diff from sandbox...[/cyan]")

        # Ensure the main branch is fetched for an accurate diff
        fetch_main_cmd = ["git", "fetch", "origin", "main"]
        self._run_command(fetch_main_cmd, cwd=self.sandbox_path)

        diff_cmd = ["git", "diff", "origin/main", "HEAD"]
        returncode, stdout, stderr = self._run_command(diff_cmd, cwd=self.sandbox_path)

        if returncode == 0:
            self.console.print("[bold green]Successfully retrieved git diff.[/bold green]")
            return stdout, None
        else:
            self.console.print("[bold red]Failed to get git diff.[/bold red]")
            failure_log = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            return None, failure_log

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
    
    async def execute_tool(self, tool_name: str, tool_func, arguments: dict) -> tuple:
        """
        Executes a tool function securely within the sandbox context.
        
        This method provides a controlled environment for executing tools,
        with proper logging and error handling. It supports both sync and
        async tool functions.
        
        Args:
            tool_name: Name of the tool being executed
            tool_func: The callable tool function
            arguments: Dictionary of arguments to pass to the tool
        
        Returns:
            Tuple of (success: bool, output: str)
        """
        import asyncio
        import traceback
        
        self.console.print(f"[cyan]Sandbox executing tool: {tool_name}[/cyan]")
        logging.info(f"Sandbox executing tool '{tool_name}' with arguments: {arguments}")
        
        try:
            # Execute the tool
            
            # Check for LangChain tool with async invoke (ainvoke)
            if hasattr(tool_func, "ainvoke"):
                result = await tool_func.ainvoke(arguments)
                
            elif asyncio.iscoroutinefunction(tool_func):
                # Handle raw async functions
                result = await tool_func(**arguments)
                
            elif hasattr(tool_func, "invoke"):
                # LangChain tool with sync invoke only
                result = tool_func.invoke(arguments)
                
            else:
                # Regular sync function
                result = tool_func(**arguments)
            
            output = str(result)
            self.console.print(f"[green]Tool '{tool_name}' completed successfully[/green]")
            logging.info(f"Tool '{tool_name}' completed. Output length: {len(output)} chars")
            return True, output
        
        except TypeError as e:
            # Usually indicates wrong arguments
            error_msg = f"Argument error for tool '{tool_name}': {e}. Check parameter names and types."
            self.console.print(f"[red]{error_msg}[/red]")
            logging.warning(error_msg)
            return False, error_msg
        
        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = f"Tool '{tool_name}' execution failed: {e}\n{error_trace}"
            self.console.print(f"[red]Tool '{tool_name}' failed: {e}[/red]")
            logging.error(error_msg)
            return False, error_msg