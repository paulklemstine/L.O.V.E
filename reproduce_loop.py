import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

from core.deep_agent_engine import _recover_json

# Test case that causes infinite loop
invalid_json = '{"key": unquoted_value}'

try:
    print(f"Testing recovery with: {invalid_json}")
    result = _recover_json(invalid_json)
    print(f"Result: {result}")
except Exception as e:
    print(f"Caught expected exception: {e}")
    print("Test PASSED: Infinite loop avoided.")

