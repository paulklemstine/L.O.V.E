
import os
import io
import sys
from rich.text import Text
from core.art_utils import save_ansi_art

# Redirect stdout to capture it
captured_output = io.StringIO()
sys.stdout = captured_output

try:
    print("START VERIFICATION")
    # Test Content
    test_content = Text("THIS IS A TEST ECHO")
    
    # Call the function
    save_ansi_art(test_content, "verify_test", output_dir="temp_art_verify")
    
    print("END VERIFICATION")
finally:
    # Restore stdout
    sys.stdout = sys.__stdout__

output = captured_output.getvalue()

# Check results
print(f"Captured Output Length: {len(output)}")
if "THIS IS A TEST ECHO" in output:
    print("FAILURE: Console echo detected!")
    exit(1)
else:
    print("SUCCESS: No console echo detected.")
    exit(0)
