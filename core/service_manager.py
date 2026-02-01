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

    # State file to track model failures/success
    STATE_FILE = "state/vllm_state.json"

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
            
    def _load_state(self):
        """Loads vLLM state from JSON file."""
        import json
        state_path = self.root_dir / self.STATE_FILE
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self, state):
        """Saves vLLM state to JSON file."""
        import json
        state_path = self.root_dir / self.STATE_FILE
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save vLLM state: {e}")

    def _get_model_tiers(self):
        """Returns ordered list of model tiers (smallest to largest)."""
        # Prioritizing REASONING models (OpenCompass Leaders) where possible.
        # Micro/Small -> Qwen 2.5 (Best general reasoning at this size)
        # Medium -> DeepSeek-R1-Distill-Qwen-14B (Reasoning SOTA)
        # Massive -> DeepSeek-R1-Distill-Llama-70B (Reasoning SOTA)
        return [
             # Tier 1: Micro (< 6GB) -> Qwen2.5-3B (Strongest <3B reasoner)
            (0, "Qwen/Qwen2.5-3B-Instruct-AWQ"), 
            (4000, "Qwen/Qwen2.5-3B-Instruct-AWQ"),
            
            # Tier 2: Small (6-16GB) -> Qwen2.5-7B (Strongest 7B reasoner)
            (6000, "Qwen/Qwen2.5-7B-Instruct-AWQ"),
            
            # Tier 3: Medium (16-24GB) -> DeepSeek-R1-Distill-Qwen-14B (Reasoning Specialist)
            # Fits in ~10-12GB (AWQ), massive math/logic gains over base models
            (16000, "Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ"),
            
            # Tier 4: Large (>24GB) -> DeepSeek-R1-Distill-Qwen-32B or Qwen2.5-32B
            # Using Qwen2.5-32B as the stable high-reasoning fallback if R1-32B isn't AWQ'd yet
            (24000, "Qwen/Qwen2.5-32B-Instruct-AWQ"),
             
            # Tier 5: Massive (>40GB) -> DeepSeek R1 Distill Llama 70B (The King)
            (40000, "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
        ]

    def _select_model(self, vram_mb):
        """Selects the best model based on VRAM and past failure state."""
        tiers = self._get_model_tiers()
        
        # 1. Filter tiers that theoretically fit
        # We assume 'min_vram' is the hard floor.
        candidates = [t for t in tiers if vram_mb >= t[0]]
        
        if not candidates:
            # Should not happen as first tier starts at 0, but fallback to smallest
            logger.warning(f"VRAM {vram_mb}MB seems extremely low. Defaulting to smallest model.")
            return tiers[0][1]
            
        # 2. Sort by size (descending) - we want the biggest that fits
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Check for "Strikes" (previous crashes)
        state = self._load_state()
        strikes = state.get("strikes", 0)
        
        if strikes > 0:
            logger.warning(f"⚠️ Detected {strikes} previous vLLM failure(s). Downgrading model selection.")
            
        # Select candidate index based on strikes
        # 0 = Best fit (Largest)
        # 1 = Next best
        # etc.
        selection_idx = min(strikes, len(candidates) - 1)
        selected_model = candidates[selection_idx][1]
        
        if selection_idx > 0:
             logger.info(f"📉 Downgraded from {candidates[0][1]} to {selected_model} due to stability history.")
             
        return selected_model

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

        # --- Dynamic Model Selection ---
        if model_name is None:  # Only auto-select if not overridden
            model_name = self._select_model(vram_mb)
            
        print(f"🚀 Starting vLLM server with model: {model_name}...")
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
        # vram_mb was already fetched at start of method
        
        if vram_mb is not None:
            print(f"   Detected VRAM: {vram_mb} MB")
            print("   ✨ Unleashing full context window (User Override).")
            
            if vram_mb < 24000:
                print("   ⚠️ Consumer GPU detected. Enforcing eager execution for stability (fixes CUDA graph hangs).")
                cmd.extend(["--enforce-eager"])
        else:
             print("   ⚠️ Unknown VRAM (nvidia-smi failed).")

        # Start process
        try:
            # We redirect stdout/stderr to a log file
            log_file = open(self.root_dir / "logs" / "vllm.log", "w")
            
            # Pass absolute venv path to script to avoid relative path issues
            env = os.environ.copy()
            env["VLLM_VENV_PATH"] = str(self.root_dir / ".venv_vllm")
            
            # Fix for max_model_len error (override model config capability)
            env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            
            self.vllm_process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            print(f"   PID: {self.vllm_process.pid}")
            print("   Waiting for vLLM to become ready (this may take a minute)...")
            return self.wait_for_vllm(model_name=model_name)
            
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

    def wait_for_vllm(self, model_name, timeout=600):
        # Wait for vLLM to become ready
        print("⏳ Waiting for vLLM API to respond...")
        for i in range(timeout):  # Wait up to `timeout` seconds
            if self.is_vllm_healthy():
                print(f"✅ vLLM is online! (Model: {model_name})")
                
                # Success! Reset strikes
                self._save_state({"strikes": 0, "last_success_model": model_name})
                return True
            time.sleep(1)
            
            # Check if process died
            if self.vllm_process and self.vllm_process.poll() is not None:
                # Capture output
                # Note: communicate() will wait for process to terminate if not already
                # Since poll() is not None, it has terminated.
                stdout, stderr = self.vllm_process.communicate()
                print(f"❌ vLLM process died unexpectedly! Exit code: {self.vllm_process.returncode}")
                if stdout: print(f"STDOUT: {stdout.decode()}")
                if stderr: print(f"STDERR: {stderr.decode()}")
                
                # Record Strike
                state = self._load_state()
                state["strikes"] = state.get("strikes", 0) + 1
                self._save_state(state)
                
                return False
            
            # Simple loading animation/feedback
            if i % 5 == 0:
                print(f"   Waiting... ({i+1}s)")

        print("❌ vLLM timed out waiting for health check.")
        # Timeout is also a strike (maybe hung on model load)
        state = self._load_state()
        state["strikes"] = state.get("strikes", 0) + 1
        self._save_state(state)
        
        return False
