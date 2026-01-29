
import sys
import os
import pytest

# Add parent directory to path to import tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tool module dynamically
import importlib.util
tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"dummy.py")
spec = importlib.util.spec_from_file_location(f"dummy", tool_path)
module = importlib.util.module_from_spec(spec)
sys.modules[f"dummy"] = module
spec.loader.exec_module(module)
from dummy import dummy

import pytest

def test_tool():
    assert True