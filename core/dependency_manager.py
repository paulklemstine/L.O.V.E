import sys
import subprocess
import shutil
import os
import logging
import platform
import urllib.request
import tempfile
import importlib

# Configure a basic logger for this module
logger = logging.getLogger(__name__)

def get_pip_executable():
    """
    Determines the correct pip command to use, returning it as a list.
    Prefers using the interpreter's own pip module for robustness.
    If pip is not found, it attempts to install it using ensurepip or get-pip.py.
    """
    # 1. Try sys.executable -m pip
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return [sys.executable, '-m', 'pip']
    except subprocess.CalledProcessError:
        pass

    # 2. Check PATH for pip3 or pip
    if shutil.which('pip3'):
        return ['pip3']
    elif shutil.which('pip'):
        return ['pip']

    # 3. Try ensurepip
    print("WARN: 'pip' not found. Attempting to install it using 'ensurepip'...")
    try:
        import ensurepip
        ensurepip.bootstrap()
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', '--version'],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            print("Successfully installed 'pip' using 'ensurepip'.")
            return [sys.executable, '-m', 'pip']
        except subprocess.CalledProcessError:
            pass
    except (ImportError, Exception) as e:
        print(f"WARN: Failed to bootstrap 'pip' with ensurepip: {e}")

    # 4. Try get-pip.py (Download and run)
    print("Attempting to install 'pip' using 'get-pip.py'...")
    try:
        url = "https://bootstrap.pypa.io/get-pip.py"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            print(f"Downloading {url} to {tmp_file.name}...")
            urllib.request.urlretrieve(url, tmp_file.name)
            tmp_path = tmp_file.name

        print(f"Running {tmp_path}...")
        subprocess.check_call([sys.executable, tmp_path, "--break-system-packages"])
        
        # Cleanup
        os.unlink(tmp_path)

        # Check again
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        print("Successfully installed 'pip' using 'get-pip.py'.")
        return [sys.executable, '-m', 'pip']

    except Exception as e:
        print(f"CRITICAL: Failed to install 'pip' using 'get-pip.py': {e}")

    # 5. Try system package manager (Linux only)
    if platform.system() == "Linux" and "TERMUX_VERSION" not in os.environ:
        print("Attempting to install 'python3-pip' via apt-get...")
        try:
            subprocess.check_call("sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip", shell=True)
            if shutil.which('pip3'):
                return ['pip3']
            elif shutil.which('pip'):
                return ['pip']
            # Check sys.executable again
            subprocess.check_call([sys.executable, '-m', 'pip', '--version'],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            return [sys.executable, '-m', 'pip']
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL: Failed to install 'python3-pip' via 'apt-get': {e}")

    return None

def install_package(package_name, upgrade=False, break_system_packages=True):
    """
    Installs a single package using the detected pip executable.
    Returns True if successful, False otherwise.
    """
    # Check if already installed
    try:
        importlib.import_module(package_name)
        # Some packages have different import names than install names (e.g. pyyaml -> yaml)
        # But for auto-installing missing imports, usually the name matches or we handle it caller side.
        # However, for robustness, we might just try install if import failed before calling this.
    except ImportError:
        pass

    print(f"Installing dependency: {package_name}...")
    pip_cmd = get_pip_executable()
    if not pip_cmd:
        print(f"ERROR: Could not find or install 'pip'. Cannot install '{package_name}'.")
        return False

    cmd = pip_cmd + ['install', package_name]
    if upgrade:
        cmd.append('--upgrade')
    if break_system_packages:
        cmd.append('--break-system-packages')

    try:
        subprocess.check_call(cmd)
        print(f"Successfully installed '{package_name}'.")
        
        # Force a refresh of sys.path to include potentially new user-site directories
        import site
        import importlib
        try:
            from importlib import reload
        except ImportError:
            try:
                from imp import reload
            except ImportError:
                pass

        # Reloading site module can sometimes help
        reload(site)
        
        # Explicitly add user site packages if missing
        # This handles cases where --user install happened (implicit or explicit)
        # and the path wasn't in sys.path at startup.
        if hasattr(site, 'getusersitepackages'):
            user_site = site.getusersitepackages()
            if isinstance(user_site, str):
                if user_site not in sys.path:
                    sys.path.append(user_site)
            elif isinstance(user_site, list):
                 for path in user_site:
                    if path not in sys.path:
                        sys.path.append(path)
        
        # Invalidate caches to see the new module
        importlib.invalidate_caches()
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install '{package_name}'. Reason: {e}")
        return False
