import os
import time
import subprocess
import requests
import logging
import json
from pathlib import Path
from typing import Optional

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
        self.config_file = self.root_dir / ".vllm_config"

    def ensure_vllm_setup(self):
        """Checks if vLLM is available (system or venv)."""
        # 1. Check if vLLM is already installed in the current environment (e.g. Colab system env)
        try:
            import vllm
            # Deep check: Import a module that likely triggers C++ extensions/CTypes loading
            # This detects ABI mismatches that wouldn't fail on top-level import
            from vllm.engine.arg_utils import EngineArgs
            
            # Stricter checks for dependencies known to cause C++ ABI issues (OSError: undefined symbol)
            import flashinfer
            import torch_c_dlpack_ext
            
            logger.info("✅ vLLM found in current environment (deep check passed). Skipping venv setup.")
            self.use_system_vllm = True
            return True
        except Exception as e:
            logger.warning(f"⚠️ System vLLM issue detected (falling back to managed venv): {e}")
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

    def load_config(self) -> dict:
        """Loads vLLM configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load persistence config: {e}")
        return {}

    def save_config(self, config: dict):
        """Saves vLLM configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save persistence config: {e}")

    def discover_best_model(self, vram_mb: Optional[int]) -> Optional[str]:
        """
        Uses ModelSelector and LeaderboardFetcher to find the best model.
        """
        print("🔍 Searching for the best available model on Leaderboard...")
        try:
            from core.model_selector import ModelSelector
            selector = ModelSelector()
            candidates = selector.select_best_models(vram_mb=vram_mb)
            
            if not candidates:
                print("⚠️ No suitable models found on leaderboard.")
                return None
            
            print(f"✨ Found {len(candidates)} candidates. Top 3:")
            for i, c in enumerate(candidates[:3]):
                print(f" {i+1}. {c.name} (ID: {c.repo_id or 'N/A'}, Score: {c.score}, Params: {c.params_b})")
                
            return candidates
        except Exception as e:
            logger.error(f"Error during model discovery: {e}")
            return None

    def start_vllm(self, model_name=None, gpu_memory_utilization=None):
        """Starts the vLLM server in a background subprocess."""
        if self.is_colab() and self.SKIP_VLLM_IN_COLAB:
            print("✨ Running in Google Colab. Skipping local vLLM startup.")
            print("   Assume Colab environment handles model serving or API access.")
            return True

        # Check for GPU/HPU presence before attempting setup
        vram_mb = self.get_total_vram_mb()
        if vram_mb is None:
            print("⚠️ No GPU/HPU detected (nvidia-smi failed or no device found).")
            print("   Skipping vLLM startup as it requires a GPU.")
            return True

        if gpu_memory_utilization is None:
            # Check env var, default to 0.6
            gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.6"))

        if self.is_vllm_healthy():
            print("✅ vLLM is already running and healthy.")
            return True

        if not self.ensure_vllm_setup():
            return False
            
        print("\n" + "="*50)
        print("🚀  INITIALIZING VLLM STARTUP SEQUENCE")
        print("="*50 + "\n")
        
        # Reset log file for this session
        with open(self.root_dir / "logs" / "vllm.log", "w") as f:
            f.write(f"--- VLLM Session Started at {time.ctime()} ---\n")
            
        # --- Model Selection Logic ---
        config = self.load_config()
        
        # Determine candidates
        candidate_queue = []
        
        if model_name:
            # Explicit override
            candidate_queue.append(model_name)
        elif "model_name" in config:
            # Persistence
            print(f"💾 Found saved model preference: {config['model_name']}")
            candidate_queue.append(config["model_name"])
        else:
            # Smart Discovery
            candidates = self.discover_best_model(vram_mb)
            if candidates:
                # Try the top models
                # For safety, let's limit to top 5 to avoid infinite loops if all fail
                # Use repo_id if available, otherwise name
                candidate_queue.extend([c.repo_id or c.name for c in candidates[:5]])
                
                # Filter out None values in case repo_id is missing and name is not useful? 
                # Ideally candidates should be valid.
                candidate_queue = [c for c in candidate_queue if c]
            else:
                # Fallback default
                candidate_queue.append("Qwen/Qwen2.5-0.5B-Instruct") # Safer fallback with chat template

        # Try to start models in order
        for idx, candidate in enumerate(candidate_queue):
            print(f"🚀 Attempting to start vLLM with model: {candidate} ({idx+1}/{len(candidate_queue)})")
            
            # Attempt 1: Default/Native settings
            if self._launch_process(candidate, gpu_memory_utilization, vram_mb):
                print(f"✅ Successfully started {candidate}")
                if candidate != config.get("model_name"):
                    self.save_config({"model_name": candidate})
                return True
            
            print(f"⚠️ First attempt for {candidate} failed.")
            
            # Attempt 2: Retry with reduced context window (16384)
            # This helps if the native context (e.g. 40k, 128k) causes OOM
            print(f"🔄 Retrying {candidate} with reduced context window (16384)...")
            if self._launch_process(candidate, gpu_memory_utilization, vram_mb, max_model_len=16384):
                 print(f"✅ Successfully started {candidate} (Reduced Context)")
                 if candidate != config.get("model_name"):
                    self.save_config({"model_name": candidate})
                 return True
            
            # Attempt 3: Retry with even smaller context (8192)
            print(f"🔄 Retrying {candidate} with reduced context window (8192)...")
            if self._launch_process(candidate, gpu_memory_utilization, vram_mb, max_model_len=8192):
                 print(f"✅ Successfully started {candidate} (Reduced Context 8192)")
                 if candidate != config.get("model_name"):
                    self.save_config({"model_name": candidate})
                 return True
            
            print(f"❌ Failed to start {candidate} even with reduced context. Trying next...")
            self.stop_vllm() # Cleanup any partial state
                
        print("❌ All model candidates failed to start.")
        return False

    def _launch_process(self, model_name, gpu_memory_utilization, vram_mb, max_model_len=None):
        """Internal helper to launch the process and wait for ready."""
        start_script = self.scripts_dir / "start_vllm.sh"
        cmd = ["bash", str(start_script)]
        
        if not getattr(self, 'use_system_vllm', False):
             venv_path = str(self.root_dir / ".venv_vllm")
             cmd.extend(["--venv", venv_path])
        else:
             print("   Using system-installed vLLM.")
        
        # Normalize model name (remove extra spaces often in JSON)
        model_name = model_name.strip()
        cmd.extend(["--model", model_name])
        
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        
        if vram_mb is not None:
            if vram_mb < 24000:
                 print("   ⚠️ Consumer GPU detected (< 24GB).")
            else:
                 print("   ✨ High VRAM detected.")

        # Enable checking for custom code (needed for Qwen and others)
        cmd.extend(["--trust-remote-code"])
        
        # Max Model Len Override
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Start process
        try:
            # Open in append mode so we don't lose logs from previous attempts in this session
            log_file = open(self.root_dir / "logs" / "vllm.log", "a")
            
            # Add separator for this attempt
            mode_msg = f" (Max Context: {max_model_len})" if max_model_len else " (Native Context)"
            log_file.write(f"\n--- Attempting launch of {model_name}{mode_msg} ---\n")
            log_file.flush()
            
            env = os.environ.copy()
            env["VLLM_VENV_PATH"] = str(self.root_dir / ".venv_vllm")
            env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            
            self.vllm_process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            print(f"   PID: {self.vllm_process.pid}")
            print("   Waiting for vLLM to become ready...")
            
            if self.wait_for_vllm(timeout=600): # 10 mins timeout per candidate
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Launch exception: {e}")
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

    def wait_for_vllm(self, timeout=600):
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
