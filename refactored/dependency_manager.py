import os
import sys
import subprocess
import shutil
import platform
import json
import logging
import importlib.metadata
from typing import List, Dict, Any, Callable

# Assuming love_state is passed as an argument where needed.

STATE_FILE = "love_state.json"

def _temp_log_event(message: str, level: str = "INFO") -> None:
    """A temporary logger that writes directly to the logging module."""
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    else:
        logging.critical(message)

def _temp_save_state(love_state: Dict[str, Any]) -> None:
    """A temporary state saver that writes directly to the state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(love_state, f, indent=4)
    except (IOError, TypeError) as e:
        logging.critical(f"CRITICAL: Could not save state during dependency check: {e}")

def is_dependency_met(love_state: Dict[str, Any], dependency_name: str) -> bool:
    """Checks if a dependency has been marked as met in the state."""
    return love_state.get("dependency_tracker", {}).get(dependency_name, False)

def mark_dependency_as_met(love_state: Dict[str, Any], dependency_name: str) -> None:
    """Marks a dependency as met in the state and saves the state."""
    love_state.setdefault("dependency_tracker", {})[dependency_name] = True
    _temp_save_state(love_state)
    _temp_log_event(f"Dependency met and recorded: {dependency_name}", "INFO")

def _install_system_packages(love_state: Dict[str, Any]) -> None:
    """
    Ensures the environment is correctly configured with necessary paths and aliases.
    Also checks/installs system packages like build-essential, cmake, curl, and docker.
    """
    if is_dependency_met(love_state, "system_packages"):
        print("System packages already installed (tracker). Skipping.")
        return

    home_dir: str = os.path.expanduser("~")
    local_bin: str = os.path.join(home_dir, ".local", "bin")

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
            python_symlink: str = os.path.join(local_bin, "python")
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
        packages: List[str] = []
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
                cmd: str = f"sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q {' '.join(packages)}"
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
                current_user: str = getpass.getuser()
                subprocess.check_call(f"sudo usermod -aG docker {current_user}", shell=True)
                print("Successfully installed 'docker'.")
                print(f"IMPORTANT: You need to log out and back in for Docker group membership to take effect.")
            except subprocess.CalledProcessError as e:
                print(f"WARN: Failed to install 'docker'. Error: {e}")
                print("You may need to install Docker Desktop for Windows and enable WSL integration manually.")

    mark_dependency_as_met(love_state, "system_packages")

def _get_pip_executable() -> List[str]:
    """
    Determines the correct pip command to use, returning it as a list.
    """
    from core.dependency_manager import get_pip_executable
    return get_pip_executable()

def _is_package_installed(req_str: str) -> bool:
    """Checks if a package specified by a requirement string is installed."""
    try:
        from packaging.requirements import Requirement
        req = Requirement(req_str)
        try:
             installed_version: str = importlib.metadata.version(req.name)
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

def _install_requirements_file(requirements_path: str, tracker_prefix: str) -> None:
    """
    Parses a requirements file and installs packages using pip.
    """
    if not os.path.exists(requirements_path):
        print(f"WARN: Requirements file not found at '{requirements_path}'. Skipping.")
        logging.warning(f"Requirements file not found at '{requirements_path}'.")
        return

    print(f"Installing dependencies from {requirements_path}...")
    pip_executable: List[str] = _get_pip_executable()
    if not pip_executable:
        print("ERROR: Could not find 'pip' or 'pip3'. Please ensure pip is installed.")
        logging.error("Could not find 'pip' or 'pip3'.")
        return

    try:
        install_command: List[str] = pip_executable + ['install', '-r', requirements_path, '--break-system-packages', '--no-input', '--disable-pip-version-check', '-v']
        subprocess.check_call(install_command)
        print(f"Successfully installed dependencies from {requirements_path}.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies from {requirements_path}. Reason: {e}")
        logging.error(f"Failed to install dependencies from {requirements_path}: {e}")
        sys.exit(1)

def _install_python_requirements(love_state: Dict[str, Any]) -> None:
    """Installs Python packages from requirements.txt and OS-specific dependencies."""
    print("Checking core Python packages...")

    try:
        from core.dependency_manager import install_package
        print("Ensuring typing_extensions is installed...")
        install_package("typing_extensions")
    except Exception as e:
        print(f"WARN: Failed to upgrade typing_extensions: {e}")

    pip_executable: List[str] = _get_pip_executable()
    if pip_executable:
        if not _is_package_installed("pip-tools"):
            try:
                subprocess.check_call(pip_executable + ['install', 'pip-tools', '--break-system-packages', '--no-input'])
            except subprocess.CalledProcessError:
                pass

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
                 sys.exit(1)
        else:
             print("WARN: No requirements file found.")

    if platform.system() == "Linux":
        print("Checking for torch-c-dlpack-ext optimization...")
        if not is_dependency_met(love_state, "torch_c_dlpack_ext_installed"):
            if not _is_package_installed("torch-c-dlpack-ext"):
                print("Installing torch-c-dlpack-ext for performance...")
                pip_executable = _get_pip_executable()
                if pip_executable:
                    try:
                        subprocess.check_call(pip_executable + ['install', 'torch-c-dlpack-ext', '--break-system-packages'])
                        print("Successfully installed torch-c-dlpack-ext.")
                        mark_dependency_as_met(love_state, "torch_c_dlpack_ext_installed")
                    except subprocess.CalledProcessError as e:
                        print(f"WARN: Failed to install torch-c-dlpack-ext. Performance might be suboptimal. Reason: {e}")
                        logging.warning(f"Failed to install torch-c-dlpack-ext: {e}")
                else:
                    print("ERROR: Could not find pip to install torch-c-dlpack-ext.")
            else:
                print("torch-c-dlpack-ext is already installed.")
                mark_dependency_as_met(love_state, "torch_c_dlpack_ext_installed")

    if platform.system() == "Windows":
        print("Windows detected. Checking for pywin32 dependency...")
        if not is_dependency_met(love_state, "pywin32_installed"):
            if not _is_package_installed("pywin32"):
                print("Installing pywin32 for Windows...")
                pip_executable = _get_pip_executable()
                if pip_executable:
                    try:
                        subprocess.check_call(pip_executable + ['install', 'pywin32', '--break-system-packages'])
                        print("Successfully installed pywin32.")
                        mark_dependency_as_met(love_state, "pywin32_installed")
                    except subprocess.CalledProcessError as e:
                        print(f"ERROR: Failed to install pywin32. Reason: {e}")
                        logging.error(f"Failed to install pywin32: {e}")
                else:
                    print("ERROR: Could not find pip to install pywin32.")
                    logging.error("Could not find pip to install pywin32.")
            else:
                print("pywin32 is already installed.")
                mark_dependency_as_met(love_state, "pywin32_installed")

def _check_and_install_dependencies(love_state: Dict[str, Any], cleanup_gpu_processes: Callable[[], None], auto_configure_hardware: Callable[[], None], configure_llm_api_key: Callable[[], None]) -> None:
    """
    Orchestrates the installation of all dependencies.
    """
    print("--- L.O.V.E. PRE-FLIGHT CHECK ---")
    cleanup_gpu_processes()
    _install_system_packages(love_state)
    _install_python_requirements(love_state)
    auto_configure_hardware()
    configure_llm_api_key()
    print("--- PRE-FLIGHT CHECK COMPLETE ---")
