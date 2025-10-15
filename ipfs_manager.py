import os
import subprocess
import logging
import time
import shutil
import atexit
import sys
import platform

class IPFSManager:
    """
    Manages the lifecycle of a local IPFS daemon.
    - Checks for the correct IPFS version.
    - Compiles IPFS from source if it's missing or incorrect.
    - Starts and stops the daemon.
    - Ensures the daemon is terminated on script exit.
    """
    def __init__(self, console, repo_path="./.ipfs_repo", bin_path="./bin/ipfs"):
        self.console = console
        self.repo_path = os.path.abspath(repo_path)
        self.bin_path = os.path.abspath(bin_path)
        self.log_file = "ipfs.log"
        self.daemon_process = None
        self.kubo_repo_url = "https://github.com/ipfs/kubo.git"
        self.kubo_dir = os.path.abspath("./kubo")

        # Ensure the daemon is cleaned up on exit
        atexit.register(self.stop_daemon)

    def _run_command(self, command, cwd=None, env=None):
        """A helper to run shell commands and log their output."""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logging.error(f"Command '{' '.join(command)}' failed with code {process.returncode}")
                logging.error(f"STDOUT: {stdout}")
                logging.error(f"STDERR: {stderr}")
                return False, stderr
            return True, stdout
        except Exception as e:
            logging.error(f"Exception running command '{' '.join(command)}': {e}")
            return False, str(e)

    def _check_go_installed(self):
        """Checks if Go is installed and available."""
        self.console.print("[cyan]Checking for Go compiler...[/cyan]")
        if shutil.which("go"):
            self.console.print("[green]Go compiler found.[/green]")
            return True

        self.console.print("[yellow]Go compiler not found. Attempting to install...[/yellow]")
        if platform.system() == "Linux":
            try:
                subprocess.check_call("sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q golang", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.console.print("[green]Go compiler installed successfully.[/green]")
                return True
            except subprocess.CalledProcessError as e:
                self.console.print(f"[bold red]Failed to install Go: {e}[/bold red]")
                return False
        else:
            self.console.print("[bold red]Go installation not automated for this OS. Please install Go manually.[/bold red]")
            return False

    def _install_ipfs(self):
        """Clones and builds the latest IPFS binary from source."""
        if not self._check_go_installed():
            return False

        self.console.print(f"[cyan]Installing latest IPFS version from source...[/cyan]")

        # Clean up old artifacts if they exist
        if os.path.exists(self.kubo_dir):
            self.console.print(f"[yellow]Removing old kubo directory: {self.kubo_dir}[/yellow]")
            shutil.rmtree(self.kubo_dir)

        # 1. Clone the repo
        self.console.print(f"[cyan]Cloning {self.kubo_repo_url}...[/cyan]")
        success, _ = self._run_command(["git", "clone", "--depth", "1", self.kubo_repo_url, self.kubo_dir])
        if not success:
            self.console.print("[bold red]Failed to clone kubo repository.[/bold red]")
            return False

        # 2. Build the binary
        self.console.print("[cyan]Compiling IPFS binary with 'go build'...[/cyan]")
        # The output of 'go build' is 'kubo' in the cmd directory.
        build_command = ["go", "build", "-o", self.bin_path, "./cmd/ipfs"]
        success, output = self._run_command(build_command, cwd=self.kubo_dir)

        if not success:
            self.console.print(f"[bold red]Failed to compile IPFS binary. Error:\n{output}[/bold red]")
            return False

        # Create the parent directory for the binary if it doesn't exist
        os.makedirs(os.path.dirname(self.bin_path), exist_ok=True)
        # The build command now outputs directly to the bin_path

        self.console.print(f"[green]IPFS binary compiled successfully to {self.bin_path}[/green]")
        return True

    def start_daemon(self):
        """Initializes the repo and starts the IPFS daemon."""
        if not os.path.exists(self.bin_path):
            self.console.print("[bold red]IPFS binary not found. Cannot start daemon.[/bold red]")
            return False

        # Set the IPFS_PATH environment variable for the daemon process
        env = os.environ.copy()
        env['IPFS_PATH'] = self.repo_path

        # 1. Initialize the IPFS repository if it doesn't exist
        if not os.path.exists(os.path.join(self.repo_path, "config")):
            self.console.print(f"[cyan]Initializing IPFS repository at {self.repo_path}...[/cyan]")
            success, output = self._run_command([self.bin_path, "init"], env=env)
            if not success:
                self.console.print(f"[bold red]Failed to initialize IPFS repository. Error:\n{output}[/bold red]")
                return False

        # 2. Start the daemon
        self.console.print("[cyan]Starting IPFS daemon in the background...[/cyan]")
        try:
            with open(self.log_file, 'w') as log:
                # We use the absolute path to the binary to avoid PATH issues.
                self.daemon_process = subprocess.Popen(
                    [self.bin_path, "daemon", "--enable-pubsub-experiment"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT
                )

            # Wait a few seconds to see if it starts correctly
            time.sleep(5)
            if self.daemon_process.poll() is not None:
                 self.console.print(f"[bold red]IPFS daemon failed to start. Check {self.log_file} for details.[/bold red]")
                 return False

            self.console.print(f"[green]IPFS daemon started successfully. PID: {self.daemon_process.pid}[/green]")
            logging.info(f"IPFS daemon started with PID {self.daemon_process.pid}")
            return True
        except Exception as e:
            self.console.print(f"[bold red]An exception occurred while starting the IPFS daemon: {e}[/bold red]")
            logging.error(f"Exception starting IPFS daemon: {e}")
            return False

    def stop_daemon(self):
        """Stops the IPFS daemon process if it's running."""
        if self.daemon_process and self.daemon_process.poll() is None:
            self.console.print(f"[cyan]Stopping IPFS daemon (PID: {self.daemon_process.pid})...[/cyan]")
            self.daemon_process.terminate()
            try:
                self.daemon_process.wait(timeout=10)
                self.console.print("[green]IPFS daemon stopped gracefully.[/green]")
                logging.info("IPFS daemon stopped gracefully.")
            except subprocess.TimeoutExpired:
                self.console.print("[yellow]IPFS daemon did not terminate gracefully. Forcing shutdown...[/yellow]")
                self.daemon_process.kill()
                logging.warning("IPFS daemon was killed.")
        self.daemon_process = None

    def setup(self):
        """
        The main public method to set up and start the IPFS daemon.
        Orchestrates the checking, installation, and running of the daemon.
        """
        self.console.print("[bold magenta]=== IPFS Self-Management Initialized ===[/bold magenta]")
        if not os.path.exists(self.bin_path):
            self.console.print("[yellow]IPFS binary not found. Proceeding with installation.[/yellow]")
            if not self._install_ipfs():
                self.console.print("[bold red]FATAL: IPFS installation failed. Cannot continue.[/bold red]")
                sys.exit(1)
        else:
            self.console.print("[green]IPFS binary found. Skipping installation.[/green]")

        if not self.start_daemon():
            self.console.print("[bold red]FATAL: Could not start the IPFS daemon. Cannot continue.[/bold red]")
            sys.exit(1)

        self.console.print("[bold green]=== IPFS Setup Complete and Daemon Running ===[/bold green]")
        return True