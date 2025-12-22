
import os
import sys
import subprocess
import shutil
import platform
import json
import logging
import importlib.metadata
from packaging.requirements import Requirement

VRAM_MODEL_MAP = {
    4096:  {"repo_id": "TheBloke/stable-code-3b-GGUF", "filename": "stable-code-3b.Q3_K_M.gguf"},
    6140:  {"repo_id": "unsloth/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q5_K_S.gguf"},
    8192:  {"repo_id": "TheBloke/Llama-2-13B-chat-GGUF", "filename": "llama-2-13b-chat.Q4_K_M.gguf"},
    16384: {"repo_id": "TheBloke/CodeLlama-34B-Instruct-GGUF", "filename": "codellama-34b-instruct.Q4_K_M.gguf"},
    32768: {"repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "filename": "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"},
}

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

def _temp_save_state(love_state, state_file):
    """A temporary state saver that writes directly to the state file."""
    try:
        with open(state_file, 'w') as f:
            json.dump(love_state, f, indent=4)
    except (IOError, TypeError) as e:
        logging.critical(f"CRITICAL: Could not save state during dependency check: {e}")

def is_dependency_met(love_state, dependency_name):
    """Checks if a dependency has been marked as met in the state."""
    return love_state.get("dependency_tracker", {}).get(dependency_name, False)

def mark_dependency_as_met(love_state, state_file, dependency_name):
    """Marks a dependency as met in the state and saves the state."""
    love_state.setdefault("dependency_tracker", {})[dependency_name] = True
    _temp_save_state(love_state, state_file)
    _temp_log_event(f"Dependency met and recorded: {dependency_name}", "INFO")

def _install_system_packages(love_state, state_file):
    """
    Ensures the environment is correctly configured with necessary paths and aliases.
    """
    if is_dependency_met(love_state, "system_packages"):
        print("System packages already installed (tracker). Skipping.")
        return

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

    if not shutil.which("python"):
        print("'python' command not found. Creating symlink to python3 in ~/.local/bin...")
        try:
            python_symlink = os.path.join(local_bin, "python")
            if not os.path.exists(python_symlink):
                try:
                    os.symlink(sys.executable, python_symlink)
                    print(f"Created symlink: {python_symlink} -> {sys.executable}")
                except OSError as e:
                    print(f"WARN: Failed to create 'python' symlink: {e}")
        except Exception as e:
             print(f"WARN: Error checking/creating python symlink: {e}")

    if platform.system() == "Linux" and "TERMUX_VERSION" not in os.environ and shutil.which("apt-get"):
        print("Ensuring system dependencies are installed...")
        packages = []
        if not shutil.which("make") or not shutil.which("gcc"):
            packages.append("build-essential")
        if not shutil.which("cmake"):
            packages.append("cmake")
        if not shutil.which("python3-dev"):
             packages.append("python3-dev")
        if not shutil.which("curl"):
             packages.append("curl")
        if not shutil.which("git"):
             packages.append("git")

        if packages:
            try:
                print(f"Installing missing system packages: {', '.join(packages)}...")
                cmd = f"sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q {' '.join(packages)}"
                subprocess.check_call(cmd, shell=True)
                print("Successfully installed system packages.")
            except subprocess.CalledProcessError as e:
                print(f"WARN: Failed to install system packages: {e}")
                print("Builds for some dependencies may fail.")

    if platform.system() == "Linux" and "TERMUX_VERSION" not in os.environ:
        if not shutil.which("docker"):
            print("Container runtime 'docker' not found. Auto-installing...")
            print("This may take a few minutes...")
            try:
                subprocess.check_call("curl -fsSL https://get.docker.com -o /tmp/get-docker.sh", shell=True)
                subprocess.check_call("sudo sh /tmp/get-docker.sh", shell=True)
                import getpass
                current_user = getpass.getuser()
                subprocess.check_call(f"sudo usermod -aG docker {current_user}", shell=True)
                print("Successfully installed 'docker'.")
                print(f"IMPORTANT: You need to log out and back in for Docker group membership to take effect.")
            except subprocess.CalledProcessError as e:
                print(f"WARN: Failed to install 'docker'. Error: {e}")
                print("You may need to install Docker Desktop for Windows and enable WSL integration manually.")

    mark_dependency_as_met(love_state, state_file, "system_packages")

def get_pip_executable():
    """
    Determines the correct pip command to use, attempting to bootstrap it if necessary.
    """
    # 1. Prioritize using the current Python interpreter to run pip
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return [sys.executable, '-m', 'pip']
    except subprocess.CalledProcessError:
        pass  # pip is not available via the current interpreter

    # 2. Try to bootstrap pip using ensurepip
    print("WARN: 'python -m pip' not found. Attempting to bootstrap pip...")
    try:
        import ensurepip
        ensurepip.bootstrap()
        # After bootstrapping, re-run the check
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Successfully bootstrapped pip.")
        return [sys.executable, '-m', 'pip']
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"WARN: Failed to bootstrap pip: {e}")

    # 3. Fallback to checking the system PATH for 'pip3' or 'pip'
    print("WARN: Falling back to searching PATH for pip executable...")
    for pip_cmd in ['pip3', 'pip']:
        if shutil.which(pip_cmd):
            print(f"Found '{pip_cmd}' in PATH.")
            return [pip_cmd]

    # 4. If all else fails
    print("CRITICAL: Could not find or install pip. Dependency installation will fail.")
    return None

def _is_package_installed(req_str):
    """Checks if a package specified by a requirement string is installed."""
    try:
        req = Requirement(req_str)
        try:
            installed_version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            return False
        if req.specifier:
            return req.specifier.contains(installed_version, prereleases=True)
        return True
    except Exception:
        try:
            importlib.metadata.version(req_str)
            return True
        except:
            return False

def _install_requirements_file(requirements_path):
    """Parses a requirements file and installs packages."""
    if not os.path.exists(requirements_path):
        print(f"WARN: Requirements file not found at '{requirements_path}'. Skipping.")
        logging.warning(f"Requirements file not found at '{requirements_path}'.")
        return

    print(f"Installing dependencies from {requirements_path}...")
    pip_executable = get_pip_executable()
    if not pip_executable:
        print("ERROR: Could not find 'pip' or 'pip3'. Please ensure pip is installed.")
        logging.error("Could not find 'pip' or 'pip3'.")
        return

    try:
        install_command = pip_executable + ['install', '-r', requirements_path, '--break-system-packages', '--no-input', '--disable-pip-version-check', '-v']
        subprocess.check_call(install_command)
        print(f"Successfully installed dependencies from {requirements_path}.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies from {requirements_path}. Reason: {e}")
        logging.error(f"Failed to install dependencies from {requirements_path}: {e}")
        sys.exit(1)

def _install_python_requirements(love_state, state_file):
    """Installs Python packages from requirements.txt and OS-specific dependencies."""
    print("Checking core Python packages...")
    try:
        install_package("typing_extensions")
    except Exception as e:
        print(f"WARN: Failed to upgrade typing_extensions: {e}")

    pip_executable = get_pip_executable()
    if pip_executable:
        if not _is_package_installed("pip-tools"):
            try:
                subprocess.check_call(pip_executable + ['install', 'pip-tools', '--break-system-packages', '--no-input'])
            except subprocess.CalledProcessError:
                pass
        if os.path.exists("requirements.txt"):
            _install_requirements_file("requirements.txt")
        elif os.path.exists("requirements.in"):
             print("requirements.txt not found. Compiling from requirements.in...")
             try:
                 subprocess.check_call([sys.executable, '-m', 'piptools', 'compile', 'requirements.in', '-vv'])
                 if os.path.exists("requirements.txt"):
                    _install_requirements_file("requirements.txt")
             except subprocess.CalledProcessError as e:
                 print(f"CRITICAL: Failed to compile requirements.in: {e}")
                 sys.exit(1)
        else:
             print("WARN: No requirements file found.")

    if platform.system() == "Linux":
        print("Checking for torch-c-dlpack-ext optimization...")
        if not is_dependency_met(love_state, "torch_c_dlpack_ext_installed"):
            if not _is_package_installed("torch-c-dlpack-ext"):
                print("Installing torch-c-dlpack-ext for performance...")
                pip_executable = get_pip_executable()
                if pip_executable:
                    try:
                        subprocess.check_call(pip_executable + ['install', 'torch-c-dlpack-ext', '--break-system-packages'])
                        print("Successfully installed torch-c-dlpack-ext.")
                        mark_dependency_as_met(love_state, state_file, "torch_c_dlpack_ext_installed")
                    except subprocess.CalledProcessError as e:
                        print(f"WARN: Failed to install torch-c-dlpack-ext. Performance might be suboptimal. Reason: {e}")
                        logging.warning(f"Failed to install torch-c-dlpack-ext: {e}")
                else:
                    print("ERROR: Could not find pip to install torch-c-dlpack-ext.")
            else:
                print("torch-c-dlpack-ext is already installed.")
                mark_dependency_as_met(love_state, state_file, "torch_c_dlpack_ext_installed")

    if platform.system() == "Windows":
        print("Windows detected. Checking for pywin32 dependency...")
        if not is_dependency_met(love_state, "pywin32_installed"):
            if not _is_package_installed("pywin32"):
                print("Installing pywin32 for Windows...")
                pip_executable = get_pip_executable()
                if pip_executable:
                    try:
                        subprocess.check_call(pip_executable + ['install', 'pywin32', '--break-system-packages'])
                        print("Successfully installed pywin32.")
                        mark_dependency_as_met(love_state, state_file, "pywin32_installed")
                    except subprocess.CalledProcessError as e:
                        print(f"ERROR: Failed to install pywin32. Reason: {e}")
                        logging.error(f"Failed to install pywin32: {e}")
                else:
                    print("ERROR: Could not find pip to install pywin32.")
                    logging.error("Could not find pip to install pywin32.")
            else:
                print("pywin32 is already installed.")
                mark_dependency_as_met(love_state, state_file, "pywin32_installed")

def _auto_configure_hardware(love_state, state_file):
    """Detects hardware and configures the state accordingly."""
    love_state.setdefault('hardware', {})
    if love_state['hardware'].get('gpu_detected'):
        print("GPU previously detected. Skipping hardware check.")
        _temp_log_event("GPU previously detected, skipping hardware check.", "INFO")
        return

    _temp_log_event("Performing hardware auto-configuration check...", "INFO")
    print("Performing hardware auto-configuration check...")

    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        love_state['hardware']['cpu_count'] = cpu_count
        _temp_log_event(f"Detected {cpu_count} CPU cores.", "INFO")
        print(f"Detected {cpu_count} CPU cores.")
    except Exception as e:
        love_state['hardware']['cpu_count'] = 0
        _temp_log_event(f"Could not detect CPU cores: {e}", "WARNING")
        print(f"WARNING: Could not detect CPU cores: {e}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        vram_mb = int(result.stdout.strip())
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
                free_vram_mb = int(free_mem_result.stdout.strip())
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

                mark_dependency_as_met(love_state, state_file, "rust_installed")
                print("Successfully installed Rust compiler.")
                _temp_log_event("Rust compiler installed successfully.", "INFO")
            except Exception as rust_error:
                _temp_log_event(f"Failed to install Rust compiler: {rust_error}", "WARNING")
                print(f"WARNING: Failed to install Rust compiler: {rust_error}")
                print("Some Python packages may fail to build from source.")

        if not is_dependency_met(love_state, "deepagent_deps_installed"):
            print("GPU detected. Installing DeepAgent dependencies (including vLLM)...")
            pip_executable = get_pip_executable()
            if pip_executable:
                try:
                    print("Installing requirements from requirements-deepagent.txt...")
                    subprocess.check_call(pip_executable + ['install', '-r', 'requirements-deepagent.txt', '--upgrade', '--break-system-packages'])
                    mark_dependency_as_met(love_state, state_file, "deepagent_deps_installed")
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

    _temp_save_state(love_state, state_file)

def cleanup_gpu_processes():
    """Kills zombie vLLM processes."""
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

def _configure_llm_api_key():
    """Configures the API key for the `llm` command-line tool."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            if not shutil.which('llm'):
                _temp_log_event("'llm' command not found. Skipping API key configuration for now.", "INFO")
                return
            result = subprocess.run(
                ['llm', 'keys', 'list'],
                capture_output=True, text=True, check=True
            )
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
                error_message += f"Stderr: {e.stderr}"
            _temp_log_event(error_message, "WARNING")
    else:
        _temp_log_event("GEMINI_API_KEY environment variable not found. LLM functionality may be limited.", "WARNING")

def check_and_install_dependencies(love_state, state_file):
    """Orchestrates the installation of all dependencies."""
    print("--- L.O.V.E. PRE-FLIGHT CHECK ---")
    cleanup_gpu_processes()
    _install_system_packages(love_state, state_file)
    _install_python_requirements(love_state, state_file)
    _auto_configure_hardware(love_state, state_file)
    _configure_llm_api_key()
    print("--- PRE-FLIGHT CHECK COMPLETE ---")

def install_package(package_name):
    """Installs a single package using pip."""
    pip_executable = get_pip_executable()
    if not pip_executable:
        print(f"ERROR: Could not find pip to install {package_name}.")
        return False
    try:
        subprocess.check_call(pip_executable + ['install', package_name, '--break-system-packages', '--no-input'])
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install {package_name}. Reason: {e}")
        return False
