import sys
import os
import json

# Add the project root to the python path
sys.path.append(os.getcwd())

from core.deep_agent_engine import _recover_json

# Test case from user report (truncated)
invalid_json = '''ALLOWED ```json
{
  "tool": "strategize",
  "reason": "To analyze knowledge and identify strategic opportunities for improvement and talent development.",
  "command": "strategize --repository-name 'p
'''

try:
    print(f"Testing recovery with invalid prefix...")
    result = _recover_json(invalid_json)
    print(f"Result: {result}")
except Exception as e:
    print(f"Caught expected exception: {e}")
