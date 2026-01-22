#!/usr/bin/env python3
"""
run_tests.py - L.O.V.E. Version 2 Test Runner

Installs dependencies and runs the pytest test suite.

Usage:
    python run_tests.py              # Install deps and run tests
    python run_tests.py --install-only   # Just install dependencies
    python run_tests.py --test-only      # Just run tests
"""

import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    print("=" * 60)
    print("📦 Installing L.O.V.E. v2 Dependencies")
    print("=" * 60)
    
    if not requirements_path.exists():
        print(f"⚠️ requirements.txt not found at {requirements_path}")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_path),
            "--quiet"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_tests(verbose: bool = True):
    """Run the pytest test suite."""
    print("\n" + "=" * 60)
    print("🧪 Running L.O.V.E. v2 Tests")
    print("=" * 60)
    
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print(f"⚠️ Tests directory not found at {test_dir}")
        return False
    
    args = [sys.executable, "-m", "pytest", str(test_dir)]
    if verbose:
        args.append("-v")
    
    try:
        result = subprocess.run(args)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Test execution failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="L.O.V.E. v2 Test Runner")
    parser.add_argument(
        "--install-only", 
        action="store_true", 
        help="Only install dependencies"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true", 
        help="Only run tests (skip dependency installation)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose test output"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║          L.O.V.E. Version 2 - Test Runner                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    success = True
    
    if args.install_only:
        success = install_dependencies()
    elif args.test_only:
        success = run_tests(args.verbose)
    else:
        # Default: install then test
        if install_dependencies():
            success = run_tests(args.verbose)
        else:
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
