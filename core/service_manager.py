import os
import time
import subprocess
import requests
import logging
from pathlib import Path

# Setup simple logger if not already configured
logger = logging.getLogger("ServiceManager")

class ServiceManager:
    """
    Manages background services for the L.O.V.E. agent, specifically vLLM.
    """
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.scripts_dir = self.root_dir / "scripts"
        self.vllm_process = None
        self.vllm_host = "0.0.0.0"
        self.vllm_port = 8000
        self.base_url = f"http://localhost:{self.vllm_port}" # Internal check uses localhost

    def ensure_vllm_setup(self):
        """Checks if vLLM venv exists, if not, runs setup."""
        venv_path = self.root_dir / ".venv_vllm"
        if not venv_path.exists():
            print("⚠️ vLLM environment not found. Running setup script...")
            setup_script = self.scripts_dir / "setup_vllm.sh"
            try:
                subprocess.check_call(["bash", str(setup_script)], cwd=self.root_dir)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to setup vLLM: {e}")
                return False
        return True

    def start_vllm(self, model_name=None, gpu_memory_utilization=None):
        """Starts the vLLM server in a background subprocess."""
        if gpu_memory_utilization is None:
            # Check env var, default to 0.6
            gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.6"))

        if self.is_vllm_healthy():
            print("✅ vLLM is already running and healthy.")
            return True

        if not self.ensure_vllm_setup():
            return False

        print("🚀 Starting vLLM server...")
        start_script = self.scripts_dir / "start_vllm.sh"
        
        cmd = ["bash", str(start_script)]
        if model_name:
            cmd.extend(["--model", model_name])
        
        # Add common optimization and safety limits
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        cmd.extend(["--max-model-len", "8192"])  # Cap context to prevent OOM on consumer cards

        # Start process
        try:
            # We redirect stdout/stderr to a log file
            log_file = open(self.root_dir / "logs" / "vllm.log", "w")
            self.vllm_process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            print(f"   PID: {self.vllm_process.pid}")
            print("   Waiting for vLLM to become ready (this may take a minute)...")
            return self.wait_for_vllm()
            
        except Exception as e:
            print(f"❌ Failed to start vLLM: {e}")
            return False

    def stop_vllm(self):
        """Stops the vLLM process."""
        if self.vllm_process:
            print("🛑 Stopping vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
            self.vllm_process = None
            print("   vLLM stopped.")

    def is_vllm_healthy(self):
        """Checks if vLLM is responsive."""
        try:
            # Check v1/models or health
            resp = requests.get(f"{self.base_url}/health", timeout=1)
            return resp.status_code == 200
        except:
            return False

    def wait_for_vllm(self, timeout=120):
        """Polls vLLM health until timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_vllm_healthy():
                print("✅ vLLM is ready!")
                return True
            time.sleep(2)
            # Check if process died
            if self.vllm_process and self.vllm_process.poll() is not None:
                print("❌ vLLM process exited unexpectedly. Check logs/vllm.log")
                return False
            
            # Simple loading animation/feedback
            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0:
                print(f"   Waiting... ({elapsed}s)")
                
        print("❌ Timed out waiting for vLLM.")
        return False
