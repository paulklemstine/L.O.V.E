
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
    
    # Force load EVERYTHING
    try:
        from core.bluesky_agent import BlueskyAgent
        # Just to ensure it's in namespace if needed
    except:
        pass

    try:
        adapted = get_adapted_tools()
        for name, func in adapted.items():
            if name not in reg:
                reg.register(func, name=name)
    except Exception as e:
        print(f"Error loading adapted tools: {e}")
        
    try:
        ensure_registered()
    except Exception as e:
        print(f"Error loading dynamic tools: {e}")
        
    reg.refresh()
    
    output_path = "tool_list_v2.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("L.O.V.E. COMPREHENSIVE TOOL LIST\n")
        f.write("="*80 + "\n\n")
        
        # Sort by name
        sorted_tools = sorted(reg._tools.items())
        
        for name, data in sorted_tools:
            schema = data["schema"]
            desc = schema.get("description", "No description provided.")
            f.write(f"TOOL: {name}\n")
            f.write(f"DESCRIPTION: {desc}\n")
            
            params = schema.get("parameters", {}).get("properties", {})
            if params:
                f.write("PARAMETERS:\n")
                for p_name, p_info in params.items():
                    p_desc = p_info.get("description", "No description")
                    p_type = p_info.get("type", "any")
                    f.write(f"  - {p_name} ({p_type}): {p_desc}\n")
            else:
                f.write("PARAMETERS: None\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"Tool list written to {output_path}")

if __name__ == "__main__":
    list_tools()
