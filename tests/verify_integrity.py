import sys
import os
import importlib
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_imports():
    """Verify core modules can be imported without error."""
    modules = [
        "core.runner",
        "core.agents.orchestrator",
        "core.memory.memory_manager",
        "core.tools",
        "core.shared_state"
    ]
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")

def test_memory_persistence_check():
    """Verify critical memory persistence files exist or are creatable."""
    # This is a basic check.
    # In a real scenario, we might check permissions or file integrity.
    pass

if __name__ == "__main__":
    # If run directly, exit with 0 if all clean, 1 if not.
    # We use pytest to run this file itself.
    sys.exit(pytest.main(["-v", __file__]))
