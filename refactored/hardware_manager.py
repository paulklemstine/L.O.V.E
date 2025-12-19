import os
import sys
import subprocess
import shutil
import platform
import time
import json
import logging
from typing import Dict, Any

try:
    import psutil
except ImportError:
    psutil = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from refactored.dependency_manager import is_dependency_met, mark_dependency_as_met, _get_pip_executable, _temp_log_event, _temp_save_state

def _auto_configure_hardware(love_state: Dict[str, Any], VRAM_MODEL_MAP: Dict[int, Dict[str, str]]) -> None:
    """
    Detects hardware (specifically NVIDIA GPUs) and configures the state accordingly.
    """
    love_state.setdefault('hardware', {})

    if love_state['hardware'].get('gpu_detected'):
        print("GPU previously detected. Skipping hardware check.")
        _temp_log_event("GPU previously detected, skipping hardware check.", "INFO")
        return

    _temp_log_event("Performing hardware auto-configuration check...", "INFO")
    print("Performing hardware auto-configuration check...")

    if psutil:
        try:
            cpu_count: int = psutil.cpu_count(logical=True)
            love_state['hardware']['cpu_count'] = cpu_count
            _temp_log_event(f"Detected {cpu_count} CPU cores.", "INFO")
            print(f"Detected {cpu_count} CPU cores.")
        except Exception as e:
            love_state['hardware']['cpu_count'] = 0
            _temp_log_event(f"Could not detect CPU cores: {e}", "WARNING")
            print(f"WARNING: Could not detect CPU cores: {e}")
    else:
        love_state['hardware']['cpu_count'] = 0
        _temp_log_event("psutil not found, cannot detect CPU cores.", "WARNING")
        print("WARNING: psutil not found, cannot detect CPU cores.")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        vram_mb: int = int(result.stdout.strip())
        love_state['hardware']['gpu_detected'] = True
        love_state['hardware']['gpu_vram_mb'] = vram_mb
        _temp_log_event(f"NVIDIA GPU detected with {vram_mb} MB VRAM.", "CRITICAL")
        print(f"NVIDIA GPU detected with {vram_mb} MB VRAM.")

        env_utilization = os.environ.get("GPU_MEMORY_UTILIZATION")
        if env_utilization:
             try:
                 love_state['hardware']['gpu_utilization'] = float(env_utilization)
                 _temp_log_event(f"Using GPU_MEMORY_UTILIZATION from environment: {env_utilization}", "INFO")
             except ValueError:
                 love_state['hardware']['gpu_utilization'] = 0.9
                 _temp_log_event(f"Invalid GPU_MEMORY_UTILIZATION in environment. Using default: 0.9", "WARNING")
        else:
            try:
                free_mem_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                free_vram_mb: int = int(free_mem_result.stdout.strip())
                if vram_mb < 7000:
                    love_state['hardware']['gpu_utilization'] = 0.7
                    _temp_log_event(f"Detected < 7GB VRAM ({vram_mb}MB). Setting conservative utilization: 0.7", "INFO")
                else:
                    love_state['hardware']['gpu_utilization'] = 0.9
                    _temp_log_event(f"Available VRAM is {free_vram_mb}MB. Setting standard utilization: 0.9", "INFO")
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
                love_state['hardware']['gpu_utilization'] = 0.7 if vram_mb < 7000 else 0.9
                _temp_log_event(f"Could not determine free VRAM ({e}). Using default: {love_state['hardware']['gpu_utilization']}", "WARNING")

        selected_model = None
        for required_vram, model_info in sorted(VRAM_MODEL_MAP.items(), reverse=True):
            if vram_mb >= required_vram:
                selected_model = model_info
                break

        if selected_model:
            love_state['hardware']['selected_local_model'] = selected_model
            _temp_log_event(f"Selected local model {selected_model['repo_id']} for {vram_mb}MB VRAM.", "CRITICAL")
            print(f"Selected local model: {selected_model['repo_id']}")
        else:
            love_state['hardware']['selected_local_model'] = None
            _temp_log_event(f"No suitable local model found for {vram_mb}MB VRAM. CPU fallback will be used.", "WARNING")
            print(f"No suitable local model found for {vram_mb}MB VRAM. Continuing in CPU-only mode.")

        if not is_dependency_met(love_state, "rust_installed"):
            print("Installing Rust compiler (required for building tokenizers)...")
            try:
                if platform.system() == "Windows":
                    import urllib.request
                    import tempfile
                    rustup_url = "https://win.rustup.rs/x86_64"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp_file:
                        print("Downloading rustup installer...")
                        urllib.request.urlretrieve(rustup_url, tmp_file.name)
                        rustup_path = tmp_file.name
                    print("Running rustup installer...")
                    subprocess.check_call([rustup_path, "-y"])
                    os.unlink(rustup_path)
                else:
                    print("Downloading and running rustup installer...")
                    subprocess.check_call([
                        "sh", "-c",
                        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
                    ])
                home = os.path.expanduser("~")
                cargo_bin = os.path.join(home, ".cargo", "bin")
                if cargo_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
                mark_dependency_as_met(love_state, "rust_installed")
                print("Successfully installed Rust compiler.")
                _temp_log_event("Rust compiler installed successfully.", "INFO")
            except Exception as rust_error:
                _temp_log_event(f"Failed to install Rust compiler: {rust_error}", "WARNING")
                print(f"WARNING: Failed to install Rust compiler: {rust_error}")
                print("Some Python packages may fail to build from source.")

        if not is_dependency_met(love_state, "deepagent_deps_installed"):
            print("GPU detected. Installing DeepAgent dependencies (including vLLM)...")
            pip_executable = _get_pip_executable()
            if pip_executable:
                try:
                    print("Installing requirements from requirements-deepagent.txt...")
                    subprocess.check_call(pip_executable + ['install', '-r', 'requirements-deepagent.txt', '--upgrade', '--break-system-packages'])
                    mark_dependency_as_met(love_state, "deepagent_deps_installed")
                    print("Successfully installed DeepAgent dependencies.")
                except subprocess.CalledProcessError as install_error:
                    _temp_log_event(f"Failed to install DeepAgent dependencies: {install_error}", "ERROR")
                    print(f"ERROR: Failed to install DeepAgent dependencies. DeepAgent will be unavailable.")
                    love_state['hardware']['gpu_detected'] = False
            else:
                _temp_log_event("Could not find pip to install DeepAgent dependencies.", "ERROR")
                print("ERROR: Could not find pip to install DeepAgent dependencies.")
                love_state['hardware']['gpu_detected'] = False

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        love_state['hardware']['gpu_detected'] = False
        love_state['hardware']['gpu_vram_mb'] = 0
        _temp_log_event(f"No NVIDIA GPU detected ({e}). Falling back to CPU-only mode.", "INFO")
        print("No NVIDIA GPU detected. L.O.V.E. will operate in CPU-only mode.")

    _temp_save_state(love_state)

def cleanup_gpu_processes() -> None:
    """
    Kills zombie vLLM processes and clears CUDA cache if possible.
    """
    print("Checking for zombie vLLM processes...")
    _temp_log_event("Checking for zombie vLLM processes...", "INFO")
    killed_count: int = 0
    if not psutil:
        print("Warning: psutil not installed, cannot clean up GPU processes.")
        _temp_log_event("psutil not installed, cannot clean up GPU processes.", "WARNING")
        return

    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any("vllm.entrypoints.openai.api_server" in arg for arg in cmdline):
                    if proc.info.get('status') == psutil.STATUS_ZOMBIE:
                        print(f"Killing zombie vLLM process: PID {proc.info['pid']}")
                        _temp_log_event(f"Killing zombie vLLM process: PID {proc.info['pid']}", "WARNING")
                        proc.kill()
                        killed_count += 1
                    else:
                        print(f"Found active vLLM process: PID {proc.info['pid']}. Leaving it running.")
                        _temp_log_event(f"Found active vLLM process: PID {proc.info['pid']}. Leaving it running.", "INFO")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if killed_count > 0:
            print(f"Killed {killed_count} zombie vLLM process(es). Waiting for resources to release...")
            time.sleep(5)
        else:
            print("No zombie vLLM processes found.")
    except Exception as e:
        _temp_log_event(f"Error during vLLM cleanup: {e}", "WARNING")
        print(f"Warning: Error during vLLM cleanup: {e}")

    try:
        print("Scanning nvidia-smi for zombie GPU processes...")
        smi_output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if smi_output.returncode == 0 and smi_output.stdout.strip():
            pids = [int(p) for p in smi_output.stdout.strip().split('\n') if p.strip()]
            my_pid = os.getpid()
            for pid in pids:
                if pid == my_pid:
                    continue
                try:
                    proc = psutil.Process(pid)
                    proc_name = proc.name().lower()
                    cmdline = " ".join(proc.cmdline())
                    if "python" in proc_name or "vllm" in cmdline:
                        if proc.status() == psutil.STATUS_ZOMBIE:
                            print(f"Force killing zombie GPU process: {pid} ({proc_name})")
                            _temp_log_event(f"Force killing zombie GPU process: {pid} ({proc_name})", "WARNING")
                            proc.kill()
                            killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception as e:
        print(f"Error during nvidia-smi cleanup: {e}")
