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
    # Feature flag to control vLLM startup in Colab
    SKIP_VLLM_IN_COLAB = False

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.scripts_dir = self.root_dir / "scripts"
        self.vllm_process = None
        self.vllm_host = "0.0.0.0"
        self.vllm_port = 8000
        self.base_url = f"http://localhost:{self.vllm_port}" # Internal check uses localhost

    def ensure_vllm_setup(self):
        """Checks if vLLM is available (system or venv)."""
        # 1. Check if vLLM is already installed in the current environment (e.g. Colab system env)
        try:
            import vllm
            logger.info("✅ vLLM found in current environment. Skipping venv setup.")
            self.use_system_vllm = True
            return True
        except ImportError:
            self.use_system_vllm = False

        """Checks if vLLM venv exists and is valid, if not, runs setup."""
        venv_path = self.root_dir / ".venv_vllm"
        # Check for activation script to ensure venv is actually usable
        # On Windows this might be Scripts/activate, but start_vllm.sh assumes Linux structure
        # and we are primarily targeting WSL/Linux for vLLM.
        activate_path = venv_path / "bin" / "activate"
        
        if not venv_path.exists() or not activate_path.exists():
            if venv_path.exists():
                logger.warning("⚠️ vLLM environment appears corrupted or incomplete. Removing...")
                import shutil
                try:
                    shutil.rmtree(venv_path)
                except Exception as e:
                    logger.error(f"⚠️ Failed to remove corrupt venv: {e}")
            
            logger.info("⚠️ vLLM environment not found. Running setup script...")
            setup_script = self.scripts_dir / "setup_vllm.sh"
            try:
                # Ensure checking permissions
                if os.name != 'nt':
                    subprocess.check_call(["chmod", "+x", str(setup_script)])
                    
                # Run setup script and capture output to logs
                process = subprocess.Popen(
                    ["bash", str(setup_script)],
                    cwd=self.root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream output to logger
                for line in process.stdout:
                    logger.info(f"[SETUP] {line.strip()}")
                
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, ["bash", str(setup_script)])
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to setup vLLM: {e}")
                return False
        return True

    def get_total_vram_mb(self):
        """Attempts to get total VRAM in MB using nvidia-smi."""
        try:
            # Query nvidia-smi for memory.total
            # Format: 12345 (just the number because of nounits)
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            # Take the first line (first GPU)
            vram_mb = int(output.strip().split('\n')[0])
            return vram_mb
        except Exception:
            # Fallback if nvidia-smi missing or fails (e.g. CPU mode or Mac)
            return None

    def is_colab(self):
        """Checks if running in Google Colab."""
        try:
            import sys
            return 'google.colab' in sys.modules
        except ImportError:
            return False

    def start_vllm(self, model_name=None, gpu_memory_utilization=None):
        """Starts the vLLM server in a background subprocess."""
        if self.is_colab() and self.SKIP_VLLM_IN_COLAB:
            print("✨ Running in Google Colab. Skipping local vLLM startup.")
            print("   Assume Colab environment handles model serving or API access.")
            return True

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
        
        # Explicitly pass venv path ONLY if we aren't using system vLLM
        cmd = ["bash", str(start_script)]
        
        if not getattr(self, 'use_system_vllm', False):
             venv_path = str(self.root_dir / ".venv_vllm")
             cmd.extend(["--venv", venv_path])
        else:
             print("   Using system-installed vLLM.")
        
        print(f"DEBUG SERVICE_MANAGER: Launching command: {cmd}")
        
        if model_name:
            cmd.extend(["--model", model_name])
        
        # Add common optimization and safety limits
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        
        # Dynamic Context Window Config
        # If we have < 20GB VRAM (e.g. T4 16GB, Laptop 6GB), limit context to prevent OOM/Stability issues.
        # If we have > 20GB (e.g. A100, A10g), assume we can handle larger context.
        vram_mb = self.get_total_vram_mb()
        
        if vram_mb is not None:
            print(f"   Detected VRAM: {vram_mb} MB")
            if vram_mb < 20000:
                 print("   ⚠️ Consumer/T4 GPU detected (< 20GB). Limiting context window to 8192 for stability.")
                 cmd.extend(["--max-model-len", "8192"])  # Cap context for T4/Consumer GPUs
                 # Enforce eager execution for T4 stability (fixes CUDA graph capture crashes)
                 print("   ⚠️ Enforcing eager execution for stability.")
                 cmd.extend(["--enforce-eager"])
            else:
                 print("   ✨ High VRAM detected. Unleashing full context window.")
        else:
             print("   ⚠️ Unknown VRAM (nvidia-smi failed). Enforcing safe defaults.")
             cmd.extend(["--max-model-len", "8192"])
             cmd.extend(["--enforce-eager"])

        # Start process
        try:
            # We redirect stdout/stderr to a log file
            log_file = open(self.root_dir / "logs" / "vllm.log", "w")
            
            # Pass absolute venv path to script to avoid relative path issues
            env = os.environ.copy()
            env["VLLM_VENV_PATH"] = str(self.root_dir / ".venv_vllm")
            
            self.vllm_process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                env=env,
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
