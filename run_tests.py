#!/usr/bin/env python3
import subprocess
import sys
import os

def install_dependencies():
    """Installs Python and Node.js dependencies."""
    print("--- Installing Python dependencies from requirements.txt ---")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("--- Python dependencies installed successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install Python dependencies: {e}")
        sys.exit(1)

    print("--- Installing Node.js dependencies from package.json ---")
    if os.path.exists('package.json'):
        try:
            subprocess.check_call(['npm', 'install'])
            print("--- Node.js dependencies installed successfully ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: Failed to install Node.js dependencies: {e}")
            # This is not a fatal error, as not all tests might need them.
            # In the future, we could make this a fatal error if needed.
    else:
        print("--- package.json not found, skipping Node.js dependencies ---")

def run_tests():
    """Runs the pytest test suite."""
    print("--- Running pytest suite ---")
    try:
        subprocess.check_call([sys.executable, '-m', 'pytest'])
    except subprocess.CalledProcessError as e:
        print(f"--- Pytest execution failed with exit code {e.returncode} ---")
        sys.exit(e.returncode)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="L.O.V.E. Test Runner")
    parser.add_argument("--install-only", action="store_true", help="Only install dependencies and exit.")
    parser.add_argument("--test-only", action="store_true", help="Only run tests and exit (assumes dependencies are installed).")
    args = parser.parse_args()

    if args.install_only:
        install_dependencies()
    elif args.test_only:
        run_tests()
    else:
        # Default behavior: install and then run tests
        install_dependencies()
        run_tests()
