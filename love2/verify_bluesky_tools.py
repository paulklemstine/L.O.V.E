
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool_adapter import get_adapted_tools

def verify_tools():
    print("Verifying Bluesky tools...")
    tools = get_adapted_tools()
    
    bluesky_tools = [
        "bluesky_post",
        "bluesky_reply",
        "bluesky_timeline",
        "bluesky_search",
        "generate_content"
    ]
    
    missing = []
    for tool_name in bluesky_tools:
        if tool_name in tools:
            print(f"✅ Found tool: {tool_name}")
        else:
            print(f"❌ Missing tool: {tool_name}")
            missing.append(tool_name)
            
    if missing:
        print(f"\nFailed! Missing {len(missing)} tools.")
        sys.exit(1)
    else:
        print("\nSuccess! All Bluesky tools loaded.")
        sys.exit(0)

if __name__ == "__main__":
    verify_tools()
