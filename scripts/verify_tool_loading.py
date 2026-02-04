
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_tool_loading():
    print("--- Verifying Tool Loading ---")
    
    # 1. Test Tool Adapter (L.O.V.E. v1 tools)
    print("\n1. Testing ToolAdapter...")
    try:
        from core.tool_adapter import get_adapted_tools
        tools = get_adapted_tools()
        print(f"✅ ToolAdapter loaded {len(tools)} tools:")
        for name in tools:
            print(f"   - {name}")
    except Exception as e:
        print(f"❌ ToolAdapter Failed: {e}")
        import traceback
        traceback.print_exc()
        
    # 2. Test Tool Registry (DeepAgent tools)
    print("\n2. Testing ToolRegistry...")
    try:
        from core.tool_registry import get_global_registry
        registry = get_global_registry()
        # Force refresh to load custom tools
        registry.refresh()
        
        tools = registry.list_tools()
        print(f"✅ ToolRegistry loaded {len(tools)} tools:")
        for name in tools:
            print(f"   - {name}")
            
    except Exception as e:
        print(f"❌ ToolRegistry Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_tool_loading()
