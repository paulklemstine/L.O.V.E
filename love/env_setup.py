"""
Environment setup for L.O.V.E.

This module contains functions for setting up the environment,
including package installations and configuration.
"""

import os
import sys
import shutil
import platform
import subprocess
import logging
import json
import importlib.metadata

# These will be populated by the main script.
shared_state = {}
VRAM_MODEL_MAP = {}
STATE_FILE = "love_state.json"


def _temp_log_event(message, level="INFO"):
    """A temporary logger that writes directly to the logging module."""
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    else:
        logging.critical(message)


def _temp_save_state():
    """A temporary state saver that writes directly to the state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(shared_state.love_state, f, indent=4)
    except (IOError, TypeError) as e:
        # Log this critical failure to the low-level logger
        logging.critical(f"CRITICAL: Could not save state during dependency check: {e}")


def is_dependency_met(dependency_name):
    """Checks if a dependency has been marked as met in the state."""
    return shared_state.love_state.get("dependency_tracker", {}).get(dependency_name, False)


def mark_dependency_as_met(dependency_name, console=None):
    """Marks a dependency as met in the state and saves the state."""
    shared_state.love_state.setdefault("dependency_tracker", {})[dependency_name] = True
    # The console is passed optionally to avoid issues when called from threads
    # where the global console might not be initialized.
    _temp_save_state()
    _temp_log_event(f"Dependency met and recorded: {dependency_name}", "INFO")


def _install_system_packages():
    """
    Ensures the environment is correctly configured with necessary paths and aliases.
    Specifically adds ~/.local/bin to PATH and ensures 'python' points to 'python3'.
    Also checks/installs system packages like build-essential, cmake, curl, and docker.
    """
    if is_dependency_met("system_packages"):
        print("System packages already installed (tracker). Skipping.")
        return

    # 1. Ensure ~/.local/bin is in PATH
    home_dir = os.path.expanduser("~")
    local_bin = os.path.join(home_dir, ".local", "bin")

    if not os.path.exists(local_bin):
        try:
            os.makedirs(local_bin, exist_ok=True)
        except OSError as e:
            print(f"WARN: Could not create {local_bin}: {e}")

    if local_bin not in os.environ.get("PATH", ""):
        print(f"Adding {local_bin} to PATH for this session.")
        os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    # 2. Ensure 'python' command exists (required by some build scripts like fast-downward)
    if not shutil.which("python"):
        print("'python' command not found. Creating symlink to python3 in ~/.local/bin...")
        try:
            python_symlink = os.path.join(local_bin, "python")
            if not os.path.exists(python_symlink):
                # Use sys.executable to get the current python interpreter path
                try:
                    os.symlink(sys.executable, python_symlink)
                    print(f"Created symlink: {python_symlink} -> {sys.executable}")
                except OSError as e:
                    print(f"WARN: Failed to create 'python' symlink: {e}")
        except Exception as e:
            print(f"WARN: Error checking/creating python symlink: {e}")

    # 3. Install necessary system packages (build-essential, cmake, curl)
    if platform.system() == "Linux" and "TERMUX_VERSION" not in os.environ and shutil.which("apt-get"):
        print("Ensuring system dependencies are installed...")

        # Define all required packages in a single list for easier management.
        required_packages = ["build-essential", "cmake", "python3-dev", "curl", "git"]

        try:
            print(f"Updating package lists and installing: {', '.join(required_packages)}...")
            # Group the installation into a single command to be more efficient.
            # apt-get will automatically skip packages that are already installed.
            update_cmd = "sudo apt-get update -q"
            install_cmd = f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q {' '.join(required_packages)}"

            subprocess.check_call(update_cmd, shell=True)
            subprocess.check_call(install_cmd, shell=True)

            print("Successfully ensured all system packages are installed.")
        except subprocess.CalledProcessError as e:
            print(f"WARN: Failed to install system packages: {e}")
            print("Builds for some dependencies may fail.")

    # 4. Ensure Docker is installed (User Request)
    if platform.system() == "Linux" and "TERMUX_VERSION" not in os.environ:
        if not shutil.which("docker"):
            print("Container runtime 'docker' not found. Auto-installing...")
            print("This may take a few minutes...")
            try:
                # Download and run the official Docker installation script
                subprocess.check_call("curl -fsSL https://get.docker.com -o /tmp/get-docker.sh", shell=True)
                subprocess.check_call("sudo sh /tmp/get-docker.sh", shell=True)

                # Add current user to docker group
                import getpass
                current_user = getpass.getuser()
                subprocess.check_call(f"sudo usermod -aG docker {current_user}", shell=True)

                print("Successfully installed 'docker'.")
                print(f"IMPORTANT: You need to log out and back in for Docker group membership to take effect.")
            except subprocess.CalledProcessError as e:
                print(f"WARN: Failed to install 'docker'. Error: {e}")
                print("You may need to install Docker Desktop for Windows and enable WSL integration manually.")

    mark_dependency_as_met("system_packages")


def _get_pip_executable():
    """
    Determines the correct pip command to use, returning it as a list.
    Delegates to core.dependency_manager for robust handling.
    """
    from core.dependency_manager import get_pip_executable
    return get_pip_executable()


def _is_package_installed(req_str):
    """Checks if a package specified by a requirement string is installed."""
    try:
        from packaging.requirements import Requirement

        req = Requirement(req_str)
        try:
            installed_version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            return False

        if req.specifier:
            return req.specifier.contains(installed_version, prereleases=True)
        return True
    except Exception as e:
        # Fallback for simple names if packaging fails or other issues
        try:
            # fast path for just name
            importlib.metadata.version(req_str)
            return True
        except:
            return False


def _install_requirements_file(requirements_path, tracker_prefix):
    """
    Parses a requirements file and installs packages using pip.
    Now uses bulk installation for reliability and performance.
    """
    if not os.path.exists(requirements_path):
        print(f"WARN: Requirements file not found at '{requirements_path}'. Skipping.")
        logging.warning(f"Requirements file not found at '{requirements_path}'.")
        return

    print(f"Installing dependencies from {requirements_path}...")

    pip_executable = _get_pip_executable()
    if not pip_executable:
        print("ERROR: Could not find 'pip' or 'pip3'. Please ensure pip is installed.")
        logging.error("Could not find 'pip' or 'pip3'.")
        return

    try:
        # Construct the install command
        # We use --break-system-packages because this is often running in managed envs (Colab/Conda)
        # and we want to ensure we get our packages.
        install_command = pip_executable + ['install', '-r', requirements_path, '--break-system-packages', '--no-input',
                                             '--disable-pip-version-check', '-v']

        # Run pip
        subprocess.check_call(install_command)
        print(f"Successfully installed dependencies from {requirements_path}.")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies from {requirements_path}. Reason: {e}")
        logging.error(f"Failed to install dependencies from {requirements_path}: {e}")
        # Fail hard if dependencies fail - safer than continuing
        sys.exit(1)


def _install_python_requirements():
    """Installs Python packages from requirements.txt and OS-specific dependencies."""
    print("Checking core Python packages...")

    # CRITICAL FIX: Ensure typing_extensions is up-to-date for Pydantic/OpenAI
    try:
        from core.dependency_manager import install_package
        print("Ensuring typing_extensions is installed...")
        install_package("typing_extensions")
    except Exception as e:
        print(f"WARN: Failed to upgrade typing_extensions: {e}")

    pip_executable = _get_pip_executable()

    if not pip_executable:
        print("ERROR: Could not find 'pip' or 'pip3'. Some dependencies may not be installed.")
        logging.error("Could not find 'pip' or 'pip3' in _install_python_requirements.")
        return

    # Ensure pip-tools is installed
    try:
        subprocess.check_call(pip_executable + ['install', 'pip-tools', '--break-system-packages', '--no-input'])
    except subprocess.CalledProcessError:
        pass

    # Optimized Granular Install
    print("Verifying core dependencies interactively...")
    if os.path.exists("requirements.txt"):
        _install_requirements_file("requirements.txt", "core_dep_")
    elif os.path.exists("requirements.in"):
        print("requirements.txt not found. Compiling from requirements.in...")
        try:
            subprocess.check_call([sys.executable, '-m', 'piptools', 'compile', 'requirements.in', '-vv'])
            if os.path.exists("requirements.txt"):
                _install_requirements_file("requirements.txt", "core_dep_")
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL: Failed to compile requirements.in: {e}")
            # Force exit so we don't crash with missing modules later
            sys.exit(1)
    else:
        print("WARN: No requirements file found.")

    # --- Install torch-c-dlpack-ext for performance optimization ---
    # This is recommended by vLLM for better tensor allocation
    if platform.system() == "Linux":
        print("Checking for torch-c-dlpack-ext optimization...")
        try:
            subprocess.check_call(pip_executable + ['install', 'torch-c-dlpack-ext', '--break-system-packages'])
            print("Successfully installed torch-c-dlpack-ext.")
        except subprocess.CalledProcessError as e:
            print(f"WARN: Failed to install torch-c-dlpack-ext. Performance might be suboptimal. Reason: {e}")
            logging.warning(f"Failed to install torch-c-dlpack-ext: {e}")

    # --- Install Windows-specific dependencies ---
    if platform.system() == "Windows":
        print("Windows detected. Checking for pywin32 dependency...")
        try:
            subprocess.check_call(pip_executable + ['install', 'pywin32', '--break-system-packages'])
            print("Successfully installed pywin32.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install pywin32. Reason: {e}")
            logging.error(f"Failed to install pywin32: {e}")


def cleanup_gpu_processes():
    """
    Kills zombie vLLM processes.
    Uses psutil for cross-platform and reliable process termination.
    Also clears CUDA cache if possible.
    """
    import psutil
    import time

    print("Checking for zombie vLLM processes...")
    _temp_log_event("Checking for zombie vLLM processes...", "INFO")

    killed_count = 0
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any("vllm.entrypoints.openai.api_server" in arg for arg in cmdline):
                    # Check if it's running from our isolated env to avoid false positives?
                    # For now just kill any vllm server to be safe.

                    # ONLY kill if it is a zombie
                    if proc.info.get('status') == psutil.STATUS_ZOMBIE:
                        print(f"Killing zombie vLLM process: PID {proc.info['pid']}")
                        _temp_log_event(f"Killing zombie vLLM process: PID {proc.info['pid']}", "WARNING")
                        proc.kill()
                        killed_count += 1
                    else:
                        print(f"Found active vLLM process: PID {proc.info['pid']}. Leaving it running.")
                        _temp_log_event(f"Found active vLLM process: PID {proc.info['pid']}. Leaving it running.",
                                        "INFO")
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

    # --- Aggressive NVIDIA-SMI Cleanup ---
    try:
        print("Scanning nvidia-smi for zombie GPU processes...")
        # nvidia-smi --query-compute-apps=pid --format=csv,noheader
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
                    # Safety check: Don't kill important system processes if running locally
                    # In Colab/Container, it's usually fine to kill everything on GPU
                    proc_name = proc.name().lower()
                    cmdline = " ".join(proc.cmdline())

                    if "python" in proc_name or "vllm" in cmdline:
                        if proc.status() == psutil.STATUS_ZOMBIE:
                            print(f"Force killing zombie GPU process: {pid} ({proc_name})")
                            _temp_log_event(f"Force killing zombie GPU process: {pid} ({proc_name})", "WARNING")
                            proc.kill()
                            killed_count += 1
                        else:
                            # Should we log this? Maybe debug only.
                            pass

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception as e:
        print(f"Error during nvidia-smi cleanup: {e}")


def _auto_configure_hardware():
    """
    Detects hardware (specifically NVIDIA GPUs) and configures the state accordingly.
    This helps in deciding whether to install GPU-specific dependencies.
    """
    shared_state.love_state.setdefault('hardware', {})

    # Check if we've already successfully detected a GPU to avoid re-running nvidia-smi
    if shared_state.love_state.get('hardware', {}).get('gpu_detected'):
        print("GPU previously detected. Skipping hardware check.")
        _temp_log_event("GPU previously detected, skipping hardware check.", "INFO")
        return

    # Clean up GPU first to ensure accurate VRAM detection
    cleanup_gpu_processes()

    _temp_log_event("Performing hardware auto-configuration check...", "INFO")
    print("Performing hardware auto-configuration check...")

    # Add CPU core count detection
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        shared_state.love_state['hardware']['cpu_count'] = cpu_count
        _temp_log_event(f"Detected {cpu_count} CPU cores.", "INFO")
        print(f"Detected {cpu_count} CPU cores.")
    except Exception as e:
        shared_state.love_state['hardware']['cpu_count'] = 0
        _temp_log_event(f"Could not detect CPU cores: {e}", "WARNING")
        print(f"WARNING: Could not detect CPU cores: {e}")

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        vram_mb = int(result.stdout.strip())
        shared_state.love_state['hardware']['gpu_detected'] = True
        shared_state.love_state['hardware']['gpu_vram_mb'] = vram_mb
        _temp_log_event(f"NVIDIA GPU detected with {vram_mb} MB VRAM.", "CRITICAL")
        print(f"NVIDIA GPU detected with {vram_mb} MB VRAM.")

        # --- Dynamically Calculate GPU Memory Utilization ---
        # Story: Allow environment override or safer defaults for smaller cards
        env_utilization = os.environ.get("GPU_MEMORY_UTILIZATION")

        if env_utilization:
            try:
                shared_state.love_state['hardware']['gpu_utilization'] = float(env_utilization)
                _temp_log_event(f"Using GPU_MEMORY_UTILIZATION from environment: {env_utilization}", "INFO")
            except ValueError:
                shared_state.love_state['hardware']['gpu_utilization'] = 0.9
                _temp_log_event(f"Invalid GPU_MEMORY_UTILIZATION in environment. Using default: 0.9", "WARNING")
        else:
            try:
                free_mem_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                free_vram_mb = int(free_mem_result.stdout.strip())

                # Dynamic Logic
                if vram_mb < 7000:
                    # For 6GB cards, 0.95 is too aggressive. 0.7 is safer as requested.
                    shared_state.love_state['hardware']['gpu_utilization'] = 0.7
                    _temp_log_event(f"Detected < 7GB VRAM ({vram_mb}MB). Setting conservative utilization: 0.7",
                                    "INFO")
                else:
                    # For larger cards, we can be a bit more aggressive but 0.9 is usually plenty
                    shared_state.love_state['hardware']['gpu_utilization'] = 0.9
                    _temp_log_event(f"Available VRAM is {free_vram_mb}MB. Setting standard utilization: 0.9", "INFO")

            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
                # Fallback
                shared_state.love_state['hardware']['gpu_utilization'] = 0.7 if vram_mb < 7000 else 0.9
                _temp_log_event(
                    f"Could not determine free VRAM ({e}). Using default: {shared_state.love_state.get('hardware', {}).get('gpu_utilization')}",
                    "WARNING")

        # --- Select the best model based on available VRAM ---
        selected_model = None
        for required_vram, model_info in sorted(VRAM_MODEL_MAP.items(), reverse=True):
            if vram_mb >= required_vram:
                selected_model = model_info
                break

        if selected_model:
            shared_state.love_state['hardware']['selected_local_model'] = selected_model
            _temp_log_event(f"Selected local model {selected_model['repo_id']} for {vram_mb}MB VRAM.", "CRITICAL")
            print(f"Selected local model: {selected_model['repo_id']}")
        else:
            shared_state.love_state['hardware']['selected_local_model'] = None
            _temp_log_event(f"No suitable local model found for {vram_mb}MB VRAM. CPU fallback will be used.",
                            "WARNING")
            print(f"No suitable local model found for {vram_mb}MB VRAM. Continuing in CPU-only mode.")

        # Install Rust compiler if not present (needed for tokenizers and other Rust-based packages)
        if not is_dependency_met("rust_installed"):
            print("Installing Rust compiler (required for building tokenizers)...")
            try:
                if platform.system() == "Windows":
                    # On Windows, download and run rustup-init.exe
                    import urllib.request
                    import tempfile
                    rustup_url = "https://win.rustup.rs/x86_64"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp_file:
                        print("Downloading rustup installer...")
                        urllib.request.urlretrieve(rustup_url, tmp_file.name)
                        rustup_path = tmp_file.name

                    # Run rustup with -y flag for non-interactive installation
                    print("Running rustup installer...")
                    subprocess.check_call([rustup_path, "-y"])

                    # Clean up
                    import os
                    os.unlink(rustup_path)
                else:
                    # On Linux/Mac, use the official rustup script
                    print("Downloading and running rustup installer...")
                    subprocess.check_call([
                        "sh", "-c",
                        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
                    ])

                # Add cargo to PATH for current session
                import os
                home = os.path.expanduser("~")
                cargo_bin = os.path.join(home, ".cargo", "bin")
                if cargo_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"

                mark_dependency_as_met("rust_installed")
                print("Successfully installed Rust compiler.")
                _temp_log_event("Rust compiler installed successfully.", "INFO")
            except Exception as rust_error:
                _temp_log_event(f"Failed to install Rust compiler: {rust_error}", "WARNING")
                print(f"WARNING: Failed to install Rust compiler: {rust_error}")
                print("Some Python packages may fail to build from source.")

        # If GPU is detected, ensure DeepAgent dependencies are installed
        if not is_dependency_met("deepagent_deps_installed"):
            print("GPU detected. Installing DeepAgent dependencies (including vLLM)...")
            pip_executable = _get_pip_executable()
            if pip_executable:
                try:
                    # Install all DeepAgent requirements from requirements-deepagent.txt
                    print("Installing requirements from requirements-deepagent.txt...")
                    subprocess.check_call(
                        pip_executable + ['install', '-r', 'requirements-deepagent.txt', '--upgrade',
                                          '--break-system-packages'])

                    mark_dependency_as_met("deepagent_deps_installed")
                    print("Successfully installed DeepAgent dependencies.")
                except subprocess.CalledProcessError as install_error:
                    _temp_log_event(f"Failed to install DeepAgent dependencies: {install_error}", "ERROR")
                    print(f"ERROR: Failed to install DeepAgent dependencies. DeepAgent will be unavailable.")
                    shared_state.love_state['hardware']['gpu_detected'] = False  # Downgrade to CPU mode if install fails
            else:
                _temp_log_event("Could not find pip to install DeepAgent dependencies.", "ERROR")
                print("ERROR: Could not find pip to install DeepAgent dependencies.")
                shared_state.love_state['hardware']['gpu_detected'] = False


    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        shared_state.love_state['hardware']['gpu_detected'] = False
        shared_state.love_state['hardware']['gpu_vram_mb'] = 0
        _temp_log_event(f"No NVIDIA GPU detected ({e}). Falling back to CPU-only mode.", "INFO")
        print("No NVIDIA GPU detected. L.O.V.E. will operate in CPU-only mode.")

    _temp_save_state()


def _configure_llm_api_key():
    """
    Configures the API key for the `llm` command-line tool if the
    GEMINI_API_KEY environment variable is set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            # The 'llm' tool might not be installed when this is first called,
            # so we check for its existence.
            if not shutil.which('llm'):
                _temp_log_event("'llm' command not found. Skipping API key configuration for now.", "INFO")
                return

            # Check if the key is already set to avoid unnecessary subprocess calls
            result = subprocess.run(
                ['llm', 'keys', 'list'],
                capture_output=True, text=True, check=True
            )
            # A simple check to see if a google key is present.
            if 'google' in result.stdout:
                _temp_log_event("A Google API key is already configured for the llm tool.", "INFO")
                return

            _temp_log_event("Configuring Google API key for the llm tool...", "INFO")
            subprocess.run(
                ['llm', 'keys', 'set', 'google', '--value', api_key],
                check=True, capture_output=True, text=True
            )
            _temp_log_event("Successfully set Google API key for llm tool.", "INFO")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_message = f"Could not configure the 'llm' tool with Google API key. The tool might not be installed yet or another error occurred: {e}"
            if hasattr(e, 'stderr'):
                error_message += f"\nStderr: {e.stderr}"
            _temp_log_event(error_message, "WARNING")
    else:
        _temp_log_event("GEMINI_API_KEY environment variable not found. LLM functionality may be limited.", "WARNING")


def setup_environment(g_shared_state, g_VRAM_MODEL_MAP):
    """
    Orchestrates the installation of all dependencies, checking the status of each
    subsystem before attempting installation.
    """
    global shared_state, VRAM_MODEL_MAP
    shared_state = g_shared_state
    VRAM_MODEL_MAP = g_VRAM_MODEL_MAP

    # This function orchestrates the entire dependency and configuration process.
    print("--- L.O.V.E. PRE-FLIGHT CHECK ---")

    _install_system_packages()
    _install_python_requirements()
    _auto_configure_hardware()
    _configure_llm_api_key()
    print("--- PRE-FLIGHT CHECK COMPLETE ---")
