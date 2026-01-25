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

    # 5. Try system package manager (Linux only) - DISABLED to prevent sudo hangs
    # if platform.system() == "Linux" and "TERMUX_VERSION" not in os.environ:
    #     print("Attempting to install 'python3-pip' via apt-get...")
    #     # ... (code removed)
    #     print("WARN: skipping sudo install of pip")
    
    return None

    return None

def install_package(package_name, upgrade=False, break_system_packages=True):
    """
    Installs a single package using the detected pip executable.
    Returns True if successful, False otherwise.
    """
    try:
        importlib.import_module(package_name)
        # Some packages have different import names than install names (e.g. pyyaml -> yaml)
        # But for auto-installing missing imports, usually the name matches or we handle it caller side.
        if not upgrade:
            logger.info(f"Package '{package_name}' already installed and upgrade not requested.")
            return True
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
    
    cmd.append('--no-input')
    cmd.append('--disable-pip-version-check')

    try:
        subprocess.check_call(cmd)
        print(f"Successfully installed '{package_name}'.")
        
        # Force a refresh of sys.path to include potentially new user-site directories
        import site
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


# ============================================================================
# Story 2.2: Dependency Self-Management - Import Scanning & Requirements Sync
# ============================================================================

# Common mapping of import names to package names
IMPORT_TO_PACKAGE_MAP = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "magic": "python-magic",
    "websocket": "websocket-client",
    "discord": "discord.py",
    "openai": "openai",
    "anthropic": "anthropic",
    "google.generativeai": "google-generativeai",
}

# Standard library modules to exclude
STDLIB_MODULES = {
    "abc", "aifc", "argparse", "array", "ast", "asyncio", "atexit", "audioop",
    "base64", "bdb", "binascii", "binhex", "bisect", "builtins", "bz2",
    "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs",
    "codeop", "collections", "colorsys", "compileall", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
    "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib",
    "dis", "distutils", "doctest", "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob",
    "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats",
    "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random",
    "re", "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap", "threading",
    "time", "timeit", "tkinter", "token", "tokenize", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing", "typing_extensions",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", "warnings",
    "wave", "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib",
    "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "_thread",
}


def scan_imports(directory: str, exclude_dirs: list = None) -> set:
    """
    Scans all .py files in a directory for import statements.
    
    Args:
        directory: Root directory to scan
        exclude_dirs: Directories to exclude (default: __pycache__, .git, venv, etc.)
        
    Returns:
        Set of imported module names (top-level only)
    """
    import ast
    
    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", ".git", "venv", ".venv", "env", "node_modules", ".tox"]
    
    imports = set()
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if not file.endswith(".py"):
                continue
            
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name.split(".")[0]
                            imports.add(module_name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module_name = node.module.split(".")[0]
                            imports.add(module_name)
                            
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
                logger.debug(f"Failed to parse {filepath}: {e}")
    
    # Filter out standard library modules
    external_imports = imports - STDLIB_MODULES
    
    return external_imports


def parse_requirements(requirements_path: str) -> dict:
    """
    Parses requirements.txt into a dict of {package_name: version_spec}.
    
    Args:
        requirements_path: Path to requirements.txt
        
    Returns:
        Dict mapping package names to version specs (may be empty string)
    """
    requirements = {}
    
    if not os.path.exists(requirements_path):
        return requirements
    
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            
            # Handle version specifiers
            for sep in ["==", ">=", "<=", "~=", ">", "<", "!="]:
                if sep in line:
                    pkg, version = line.split(sep, 1)
                    requirements[pkg.strip().lower()] = f"{sep}{version.strip()}"
                    break
            else:
                # No version specifier
                requirements[line.lower()] = ""
    
    return requirements


def compare_requirements(actual_imports: set, requirements_path: str) -> dict:
    """
    Compares actual imports against requirements.txt.
    
    Args:
        actual_imports: Set of import names from scan_imports()
        requirements_path: Path to requirements.txt
        
    Returns:
        {missing: [...], unused: [...]}
    """
    requirements = parse_requirements(requirements_path)
    
    # Convert imports to package names
    import_packages = set()
    for imp in actual_imports:
        pkg_name = IMPORT_TO_PACKAGE_MAP.get(imp, imp).lower()
        import_packages.add(pkg_name)
    
    # Convert requirements to lowercase for comparison
    req_packages = set(requirements.keys())
    
    # Find differences
    missing = import_packages - req_packages
    unused = req_packages - import_packages
    
    return {
        "missing": sorted(missing),
        "unused": sorted(unused)
    }


def sync_requirements(
    directory: str,
    requirements_path: str = "requirements.txt",
    auto_install: bool = True,
    backup: bool = True
) -> dict:
    """
    Updates requirements.txt based on actual imports and optionally installs.
    
    Args:
        directory: Root directory to scan for imports
        requirements_path: Path to requirements.txt
        auto_install: Whether to run pip install after update
        backup: Whether to backup existing requirements.txt
        
    Returns:
        {added: [...], removed: [...], installed: bool}
    """
    result = {
        "added": [],
        "removed": [],
        "installed": False,
        "error": None
    }
    
    # Scan imports
    actual_imports = scan_imports(directory)
    logger.info(f"Found {len(actual_imports)} external imports")
    
    # Compare with requirements
    comparison = compare_requirements(actual_imports, requirements_path)
    result["added"] = comparison["missing"]
    result["removed"] = comparison["unused"]
    
    # If no changes needed
    if not result["added"] and not result["removed"]:
        logger.info("Requirements are already in sync")
        return result
    
    # Backup existing file
    if backup and os.path.exists(requirements_path):
        backup_path = f"{requirements_path}.bak"
        shutil.copy2(requirements_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
    
    # Read existing requirements
    existing = parse_requirements(requirements_path)
    
    # Add missing packages
    for pkg in result["added"]:
        existing[pkg] = ""
        logger.info(f"Adding package: {pkg}")
    
    # Remove unused packages
    for pkg in result["removed"]:
        if pkg in existing:
            del existing[pkg]
            logger.info(f"Removing package: {pkg}")
    
    # Write updated requirements
    try:
        with open(requirements_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated by dependency_manager.py\n")
            for pkg in sorted(existing.keys()):
                version = existing[pkg]
                f.write(f"{pkg}{version}\n")
        logger.info(f"Updated {requirements_path}")
    except Exception as e:
        result["error"] = str(e)
        return result
    
    # Auto-install if requested
    if auto_install:
        pip_cmd = get_pip_executable()
        if pip_cmd:
            try:
                cmd = pip_cmd + ["install", "-r", requirements_path, "--break-system-packages", "--no-input"]
                subprocess.run(cmd, check=True, capture_output=True)
                result["installed"] = True
                logger.info("Dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                result["error"] = f"Install failed: {e}"
                logger.error(f"Failed to install dependencies: {e}")
    
    return result

