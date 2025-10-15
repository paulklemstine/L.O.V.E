import os
import subprocess
import logging
import time
import shutil
import atexit
import sys
import platform
import requests
import tarfile

class IPFSManager:
    """
    Manages the lifecycle of a local IPFS daemon.
    - Downloads the latest pre-compiled IPFS binary if it's missing.
    - Starts and stops the daemon.
    - Ensures the daemon is terminated on script exit.
    """
    def __init__(self, console, repo_path="./.ipfs_repo", bin_path="./bin/ipfs"):
        self.console = console
        self.repo_path = os.path.abspath(repo_path)
        self.bin_path = os.path.abspath(bin_path)
        self.bin_dir = os.path.dirname(self.bin_path)
        self.log_file = "ipfs.log"
        self.daemon_process = None
        self.dist_url = "https://dist.ipfs.tech"

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

    def _install_dependencies(self):
        """Checks for required build dependencies."""
        dependencies = ["go", "git", "make"]
        missing = []
        for dep in dependencies:
            if not shutil.which(dep):
                missing.append(dep)

        if missing:
            self.console.print(f"[bold red]Missing required build dependencies: {', '.join(missing)}[/bold red]")
            self.console.print("[bold red]Please install them and try again.[/bold red]")
            return False
        return True

    def _install_ipfs_from_source(self):
        """Clones the Kubo repository and compiles it from source."""
        self.console.print("[cyan]Installing IPFS by compiling from source...[/cyan]")

        if not self._install_dependencies():
            return False

        # 2. Clone the Kubo repository
        kubo_source_dir = os.path.abspath("./kubo_source")
        if os.path.exists(kubo_source_dir):
            self.console.print(f"[yellow]Kubo source directory already exists at {kubo_source_dir}. Removing it.[/yellow]")
            shutil.rmtree(kubo_source_dir)

        self.console.print("[cyan]Cloning Kubo repository...[/cyan]")
        success, output = self._run_command(["git", "clone", "https://github.com/ipfs/kubo.git", kubo_source_dir])
        if not success:
            self.console.print(f"[bold red]Failed to clone Kubo repository. Error:\n{output}[/bold red]")
            return False

        # 3. Compile
        self.console.print("[cyan]Compiling Kubo... (This may take a while)[/cyan]")
        success, output = self._run_command(["make", "build"], cwd=kubo_source_dir)
        if not success:
            self.console.print(f"[bold red]Failed to compile Kubo. Error:\n{output}[/bold red]")
            return False

        # 4. Move the binary
        compiled_binary_path = os.path.join(kubo_source_dir, "cmd", "ipfs", "ipfs")
        if not os.path.exists(compiled_binary_path):
            self.console.print(f"[bold red]Compiled binary not found at {compiled_binary_path}[/bold red]")
            return False

        os.makedirs(self.bin_dir, exist_ok=True)
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)
        shutil.move(compiled_binary_path, self.bin_path)

        os.chmod(self.bin_path, 0o755)

        shutil.rmtree(kubo_source_dir)

        self.console.print(f"[green]IPFS binary compiled and installed successfully to {self.bin_path}[/green]")
        return True

    def _install_ipfs(self):
        """Installs IPFS by compiling it from source."""
        return self._install_ipfs_from_source()

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
                self.console.print("[bold red]FATAL: IPFS installation failed. IPFS features will be disabled.[/bold red]")
                return False
        else:
            self.console.print("[green]IPFS binary found. Skipping installation.[/green]")

        if not self.start_daemon():
            self.console.print("[bold red]FATAL: Could not start the IPFS daemon. IPFS features will be disabled.[/bold red]")
            return False

        self.console.print("[bold green]=== IPFS Setup Complete and Daemon Running ===[/bold green]")
        return True