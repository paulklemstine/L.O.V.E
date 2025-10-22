import subprocess
import os
import stat
import requests
import tarfile
import zipfile
import platform
import time

class NodeManager:
    """
    Manages the lifecycle of an Ethereum light node (Geth).
    - Handles installation of Geth for different OS.
    - Starts and stops the light node.
    """
    def __init__(self, data_dir="~/.ethereum_light"):
        self.data_dir = os.path.expanduser(data_dir)
        self.process = None
        self.geth_executable_path = self._get_geth_executable_path()

    def _get_geth_executable_path(self):
        """Determines the path for the Geth executable."""
        return os.path.join(os.path.dirname(__file__), "bin", "geth")

    def is_installed(self):
        """Checks if Geth is installed at the expected path."""
        return os.path.exists(self.geth_executable_path)

    def install(self):
        """
        Downloads and installs the appropriate Geth binary for the current OS.
        """
        if self.is_installed():
            print("Geth is already installed.")
            return True

        system = platform.system()
        machine = platform.machine()

        # Simplified URL logic for demonstration
        # A real implementation would need more robust URL management
        if system == "Linux":
            url = "https://gethstore.blob.core.windows.net/builds/geth-linux-amd64-1.10.26-e5eb32ac.tar.gz"
        elif system == "Darwin": # macOS
             url = "https://gethstore.blob.core.windows.net/builds/geth-darwin-amd64-1.10.26-e5eb32ac.tar.gz"
        elif system == "Windows":
            url = "https://gethstore.blob.core.windows.net/builds/geth-windows-amd64-1.10.26-e5eb32ac.zip"
        else:
            raise Exception(f"Unsupported operating system: {system}")

        print(f"Downloading Geth from {url}...")

        bin_dir = os.path.dirname(self.geth_executable_path)
        os.makedirs(bin_dir, exist_ok=True)

        download_path = os.path.join(bin_dir, os.path.basename(url))

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(download_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print("Download complete. Extracting...")

            if url.endswith(".tar.gz"):
                with tarfile.open(download_path, "r:gz") as tar:
                    # This extracts into a directory, we need to find the binary within
                    tar.extractall(path=bin_dir)
                    # Move binary to correct location
                    extracted_dir = os.path.join(bin_dir, os.path.basename(url).replace(".tar.gz", ""))
                    os.rename(os.path.join(extracted_dir, "geth"), self.geth_executable_path)

            elif url.endswith(".zip"):
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(path=bin_dir)
                    extracted_dir = os.path.join(bin_dir, os.path.basename(url).replace(".zip", ""))
                    os.rename(os.path.join(extracted_dir, "geth.exe"), self.geth_executable_path)


            print("Geth extracted.")

            # Make executable on Unix-like systems
            if system in ["Linux", "Darwin"]:
                st = os.stat(self.geth_executable_path)
                os.chmod(self.geth_executable_path, st.st_mode | stat.S_IEXEC)

            print("Geth installation successful.")
            return True
        except Exception as e:
            print(f"Failed to install Geth: {e}")
            return False
        finally:
            if os.path.exists(download_path):
                os.remove(download_path)


    def start(self):
        """
        Starts the Geth light node as a subprocess.
        """
        if not self.is_installed():
            print("Geth is not installed. Please call install() first.")
            return False

        if self.process and self.process.poll() is None:
            print("Geth is already running.")
            return True

        print("Starting Geth light node...")
        os.makedirs(self.data_dir, exist_ok=True)

        command = [
            self.geth_executable_path,
            "--syncmode", "light",
            "--datadir", self.data_dir,
            "--http",
            "--http.api", "eth,net,web3",
            "--http.port", "8545"
        ]

        try:
            # Run as a background process
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Geth process started with PID: {self.process.pid}")
            # Give it a moment to start up
            time.sleep(5)
            if self.process.poll() is not None:
                # Process terminated unexpectedly
                stdout, stderr = self.process.communicate()
                print("Geth failed to start.")
                print("Stderr:", stderr.decode())
                return False
            return True
        except Exception as e:
            print(f"Failed to start Geth: {e}")
            return False

    def stop(self):
        """
        Stops the Geth process.
        """
        if self.process and self.process.poll() is None:
            print("Stopping Geth process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("Geth process stopped.")
            except subprocess.TimeoutExpired:
                print("Geth process did not terminate gracefully. Killing.")
                self.process.kill()
        else:
            print("Geth process is not running.")