
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.tool_adapter import get_adapted_tools
    print("Successfully imported tool_adapter")
except ImportError as e:
    print(f"Failed to import tool_adapter: {e}")
    sys.exit(1)

print("Calling get_adapted_tools()...")
tools = get_adapted_tools()
print(f"Found {len(tools)} tools: {list(tools.keys())}")

if "bluesky_post" in tools:
    print("SUCCESS: bluesky_post found.")
else:
    print("FAILURE: bluesky_post NOT found.")
    
    # Try to import bluesky_agent directly to see the error
    print("\nAttempting direct import of bluesky_agent...")
    try:
        from core import bluesky_agent
        print("Direct import of bluesky_agent SUCCEEDED (unexpected).")
    except Exception as e:
        print(f"Direct import of bluesky_agent FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
