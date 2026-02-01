
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(os.getcwd())))

from core.tool_registry import get_global_registry
from core.tool_adapter import get_adapted_tools
from core.dynamic_tools import ensure_registered

def list_tools():
    reg = get_global_registry()
    
    # Load adapted tools
    try:
        adapted = get_adapted_tools()
        for name, func in adapted.items():
            if name not in reg:
                reg.register(func, name=name)
    except:
        pass
        
    # Load dynamic tools
    try:
        ensure_registered()
    except:
        pass
        
    reg.refresh()
    
    print("\n" + "="*80)
    print("L.O.V.E. AVAILABLE TOOLS")
    print("="*80 + "\n")
    
    for name, data in reg._tools.items():
        schema = data["schema"]
        desc = schema.get("description", "No description")
        print(f"[{name}]")
        print(f"  Description: {desc}")
        params = schema.get("parameters", {}).get("properties", {})
        if params:
            print(f"  Parameters: {', '.join(params.keys())}")
        else:
            print(f"  Parameters: None")
        print("-" * 40)

if __name__ == "__main__":
    list_tools()
