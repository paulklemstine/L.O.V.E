
import sys
import os

# Add project root to path (one level up from tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.tool_registry import ToolRegistry

def verify():
    registry = ToolRegistry()
    # This should now load both tools_lib and tools
    registry.load_core_tools()
    
    tools = registry.list_tools()
    print(f"Total tools loaded: {len(tools)}")
    
    required = ["evolve", "code_modifier", "post_to_bluesky"]
    missing = []
    
    for req in required:
        if req in tools:
            print(f"✅ {req} is registered.")
        else:
            print(f"❌ {req} is MISSING.")
            missing.append(req)
            
    if missing:
        print("FAIL: Some tools are missing.")
        sys.exit(1)
    else:
        print("SUCCESS: All required tools are registered.")

if __name__ == "__main__":
    verify()
