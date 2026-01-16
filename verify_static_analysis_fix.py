
import json
import logging
from unittest.mock import MagicMock, patch

# Mock the logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("core.nodes.static_analysis")

# Import the class to test
# We need to mock the imports inside static_analysis.py that we don't have or don't want to run
with patch('subprocess.run') as mock_subprocess:
    # Set up the mock to return a string error message JSON
    # This simulates the condition where radon fails or returns unexpected data
    # format that caused the crash.
    mock_subprocess.return_value.stdout = '{"file.py": "An error occurred"}'
    mock_subprocess.return_value.return_code = 0
    
    # Import locally now that we've patched external deps if needed, 
    # but here we just import the file.
    # Note: We need to make sure we can import the file. 
    # Since we are in the root, we might need to adjust sys.path or move this file.
    import sys
    import os
    sys.path.append(os.getcwd())
    
    from core.nodes.static_analysis import StaticAnalyzer
    
    analyzer = StaticAnalyzer()
    
    print("--- Testing Run Radon with Invalid JSON Structure ---")
    try:
        # This should NO LONGER raise TypeError
        result = analyzer.run_radon("dummy_file.py")
        print("SUCCESS: Function returned without crashing.")
        print(f"Result: {result}")
    except TypeError as e:
        print(f"FAILURE: Caught expected TypeError: {e}")
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {e}")

    print("\n--- Testing Run Radon with Valid JSON Structure ---")
    mock_subprocess.return_value.stdout = '{"file.py": [{"complexity": 20, "name": "complex_func", "lineno": 1}]}'
    try:
        result = analyzer.run_radon("dummy_file.py")
        print("SUCCESS: Function returned without crashing.")
        print(f"Result: {result}")
        # Verify it found the error
        if len(result['errors']) == 1:
             print("VERIFIED: Correctly identified complexity error.")
        else:
             print("FAILURE: Did not find complexity error.")
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {e}")
