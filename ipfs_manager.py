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
import re

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

    def _is_go_version_sufficient(self, min_version="1.22"):
        """Checks if the installed Go version is sufficient."""
        if not shutil.which("go"):
            self.console.print("[yellow]Go is not installed.[/yellow]")
            return False

        success, output = self._run_command(["go", "version"])
        if not success:
            self.console.print("[yellow]Could not determine Go version.[/yellow]")
            return False

        match = re.search(r"go version go(\d+\.\d+(\.\d+)?)", output)
        if not match:
            self.console.print(f"[yellow]Could not parse Go version from output: {output}[/yellow]")
            return False

        installed_version = match.group(1)
        self.console.print(f"Found Go version: {installed_version}")

        # Simple version comparison by splitting into components
        try:
            min_parts = [int(p) for p in min_version.split('.')]
            installed_parts = [int(p) for p in installed_version.split('.')]
            return installed_parts >= min_parts
        except ValueError:
            self.console.print(f"[yellow]Could not compare Go versions ('{installed_version}' vs '{min_version}'). Assuming insufficient.[/yellow]")
            return False

    def _install_go(self):
        """Downloads and installs a recent version of Go."""
        self.console.print("[cyan]Installing a recent version of Go...[/cyan]")
        GO_VERSION = "1.22.5"

        # Determine architecture
        machine_arch = platform.machine()
        if machine_arch == "x86_64":
            go_arch = "amd64"
        elif machine_arch == "aarch64":
            go_arch = "arm64"
        else:
            self.console.print(f"[bold red]Unsupported architecture for Go installation: {machine_arch}[/bold red]")
            return False

        GO_URL = f"https://go.dev/dl/go{GO_VERSION}.linux-{go_arch}.tar.gz"
        GO_TARBALL = os.path.basename(GO_URL)

        # Download
        self.console.print(f"Downloading Go from {GO_URL}...")
        try:
            response = requests.get(GO_URL, stream=True, timeout=300)
            response.raise_for_status()
            with open(GO_TARBALL, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Failed to download Go: {e}[/bold red]")
            return False

        # Install
        self.console.print("[cyan]Installing Go... (requires sudo)[/cyan]")
        self._run_command(["sudo", "rm", "-rf", "/usr/local/go"])
        success, output = self._run_command(["sudo", "tar", "-C", "/usr/local", "-xzf", GO_TARBALL])

        # Clean up tarball
        os.remove(GO_TARBALL)

        if not success:
            self.console.print(f"[bold red]Failed to extract Go tarball: {output}[/bold red]")
            return False

        # Set environment PATH for this process and any subprocesses
        go_bin_path = "/usr/local/go/bin"
        os.environ["PATH"] = f"{go_bin_path}:{os.environ.get('PATH', '')}"
        self.console.print(f"[green]Go installed and PATH configured for this session.[/green]")

        # Verify installation
        success, output = self._run_command(["go", "version"])
        if success:
            self.console.print(f"[green]Go version check successful: {output.strip()}[/green]")
            return True
        else:
            self.console.print(f"[bold red]Go version check failed after installation.[/bold red]")
            return False

    def _install_dependencies(self):
        """Checks for and installs required build dependencies (git, make, go)."""
        if platform.system() != "Linux":
            self.console.print("[bold red]Automatic dependency installation is only supported on Linux.[/bold red]")
            # Check for manual installations
            if not all(shutil.which(dep) for dep in ["git", "make", "go"]):
                self.console.print("[bold red]Please install git, make, and go and try again.[/bold red]")
                return False
            return True

        # --- Install git and make using apt ---
        apt_deps = {"git": "git", "make": "make"}
        missing_apt_deps = [pkg for dep, pkg in apt_deps.items() if not shutil.which(dep)]

        if missing_apt_deps:
            self.console.print(f"[yellow]Missing apt dependencies: {', '.join(missing_apt_deps)}. Attempting installation...[/yellow]")
            try:
                self.console.print("[cyan]Running 'apt-get update'...[/cyan]")
                update_success, update_output = self._run_command(["sudo", "apt-get", "update", "-y"])
                if not update_success:
                    # Don't fail hard on update error, sometimes it's noisy
                    self.console.print(f"[yellow]Warning: 'apt-get update' failed. Proceeding with install anyway...\n{update_output}[/yellow]")

                self.console.print(f"[cyan]Installing packages: {', '.join(missing_apt_deps)}...[/cyan]")
                install_command = ["sudo", "apt-get", "install", "-y"] + missing_apt_deps
                install_success, install_output = self._run_command(install_command)

                if not install_success:
                    self.console.print(f"[bold red]Failed to install apt dependencies. Error:\n{install_output}[/bold red]")
                    return False

                # Verify installation
                if any(not shutil.which(dep) for dep in apt_deps if dep in missing_apt_deps):
                     self.console.print(f"[bold red]Verification failed for apt packages.[/bold red]")
                     return False

            except Exception as e:
                self.console.print(f"[bold red]An error occurred during apt dependency installation: {e}[/bold red]")
                return False
        else:
            self.console.print("[green]Build dependencies 'git' and 'make' are already installed.[/green]")

        # --- Check and install Go ---
        if not self._is_go_version_sufficient():
            self.console.print("[yellow]Go version is insufficient or Go is not installed. Installing a recent version...[/yellow]")
            if not self._install_go():
                self.console.print("[bold red]Failed to install Go.[/bold red]")
                return False
        else:
            self.console.print("[green]Sufficient Go version is already installed.[/green]")

        self.console.print("[green]All build dependencies are satisfied.[/green]")
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

        # Configure ports to non-default values to avoid conflicts
        self.console.print("[cyan]Configuring IPFS API and Gateway ports to avoid conflicts...[/cyan]")
        self._run_command([self.bin_path, "config", "Addresses.API", "/ip4/127.0.0.1/tcp/5002"], env=env)
        self._run_command([self.bin_path, "config", "Addresses.Gateway", "/ip4/127.0.0.1/tcp/8888"], env=env)


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