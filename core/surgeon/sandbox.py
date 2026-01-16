
import subprocess
import logging
import os
from typing import Tuple, List

class DockerSandbox:
    def __init__(self, image_name: str = "love_surgeon_sandbox", base_dir: str = None, scratch_dir: str = None):
        """
        Args:
            image_name: Name of the docker image to use/build.
            base_dir: Root directory of the project to mount. Defaults to current working directory.
            scratch_dir: Path to scratch directory on host to mount at /scratch.
        """
        self.image_name = image_name
        # Default to the root of the repo (assuming we are running from root or finding it relative)
        # Ideally this should be passed in or reliably detected.
        # For now, we assume os.getcwd() is the project root if not specified.
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.scratch_dir = scratch_dir if scratch_dir else os.environ.get("FS_SCRATCH_PATH")
        
    def ensure_image_exists(self) -> None:
        """
        Checks if the image exists. If not, builds it using the Dockerfile in `core/surgeon/docker`.
        """
        # Check if image exists
        try:
            subprocess.run(
                ["docker", "inspect", "--type=image", self.image_name], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            logging.info(f"Docker image '{self.image_name}' found.")
        except subprocess.CalledProcessError:
            logging.info(f"Docker image '{self.image_name}' not found. Building...")
            self.build_image()

    def build_image(self) -> None:
        """
        Builds the docker image.
        """
        dockerfile_path = os.path.join(self.base_dir, "core", "surgeon", "docker", "Dockerfile")
        if not os.path.exists(dockerfile_path):
             raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        logging.info(f"Building docker image '{self.image_name}' from {self.base_dir}...")
        
        # We build from base_dir to allow COPY requirements.txt to work (assuming requirements.txt is in base_dir)
        cmd = ["docker", "build", "-t", self.image_name, "-f", dockerfile_path, "."]
        
        result = subprocess.run(
            cmd, 
            cwd=self.base_dir, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            logging.error(f"Docker build failed:\n{result.stderr}")
            raise RuntimeError(f"Failed to build docker image: {result.stderr}")
            
        logging.info("Docker image built successfully.")

    def run_command(self, command: str, timeout: int = 60, network_disabled: bool = False) -> Tuple[int, str, str]:
        """
        Runs a command inside the sandbox container.
        
        Args:
            command: Shell command to run (e.g. "python3 -m pytest tests/").
            timeout: Timeout in seconds.
            network_disabled: If True, runs with --network none.
            
        Returns:
            (exit_code, stdout, stderr)
        """
        # We mount the current codebase to /app so changes specific to the sandbox run (like tests) are visible
        # BUT we must be careful: if the command modifies files, it modifies them on the HOST.
        # Ideally, we should copy the code to a temporary volume or use a read-only mount + temp overlay?
        # Story 1.3 says "The tool writes the file back to disk" (in Story 1.2).
        # But for *execution* of tests, we might want to run strictly on what is there.
        # If we want an *ephemeral* sandbox that doesn't touch host files, we should probably COPY the code in,
        # OR mount read-only.
        # However, many tests might write artifacts (pyc, reports).
        # Current 'Sandbox' class (legacy) clones the repo to a temp dir.
        
        # New Requirement: "Container is destroyed immediately after the verification step."
        # If we mount the host dir, we risking the container changing host files (like rm -rf /app/core).
        
        # Safer approach:
        # 1. Create a volume or just let docker run on the immutable image code? 
        # But we want to test *modified* code that hasn't been committed/built into the image yet?
        # Agent modifies file on host -> runs verification in sandbox.
        # So we MUST mount the modified code.
        
        # To be safe from "rm -rf /", we should run as a non-root user inside?
        # Or just accept that "safe" means "don't destroy host system outside of mounted vol".
        # Docker default mount binds are read-write.
        
        # Let's simple mount for now, matching the requirement "mount the codebase".
        
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.base_dir}:/app",
            "-w", "/app",
        ]
        
        if self.scratch_dir:
            docker_cmd.extend(["-v", f"{self.scratch_dir}:/scratch"])

        if network_disabled:
            docker_cmd.extend(["--network", "none"])
            
        docker_cmd.extend([self.image_name, command])
        
        logging.info(f"Running in sandbox: {command}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)


def is_docker_available():
    """Check if Docker is available and running."""
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_sandbox(base_dir: str = None, scratch_dir: str = None):
    """
    Factory function to get the appropriate sandbox.
    """
    if is_docker_available():
        logging.info("Docker is available, using DockerSandbox.")
        return DockerSandbox(base_dir=base_dir, scratch_dir=scratch_dir)
    else:
        logging.info("Docker not available, falling back to LocalSandbox.")
        return LocalSandbox(base_dir=base_dir, scratch_dir=scratch_dir)


class LocalSandbox:
    def __init__(self, base_dir: str = None, scratch_dir: str = None):
        self.base_dir = base_dir if base_dir else os.getcwd()
        self.scratch_dir = scratch_dir

    def run_command(self, command: str, timeout: int = 60) -> Tuple[int, str, str]:
        """
        Runs a command in a local subprocess.
        """
        logging.info(f"Running in local sandbox: {command}")
        # If scratch_dir is provided, run the command there for isolation
        cwd = self.scratch_dir if self.scratch_dir else self.base_dir
        logging.info(f"Executing command in: {cwd}")
        try:
            # We use shell=True to allow for commands like "python3 -m pytest tests/"
            # This is less secure than shell=False, but we are running trusted code.
            # In the future we might want to parse the command and run it with shell=False
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                shell=True,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)
