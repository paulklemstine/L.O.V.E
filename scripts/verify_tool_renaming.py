
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool_registry import ToolRegistry, tool_schema

def my_original_function():
    """I am the original function."""
    pass

def verify_renaming():
    print("--- Verifying Tool Renaming ---")
    
    registry = ToolRegistry()
    
    # Register with a CUSTOM name
    target_name = "custom_tool_name"
    print(f"Registering 'my_original_function' as '{target_name}'...")
    
    registry.register(my_original_function, name=target_name)
    
    # Check schema
    schema = registry.get_schema(target_name)
    
    print(f"Schema Name: {schema['name']}")
    
    if schema['name'] == target_name:
        print("✅ SUCCESS: Schema name matches registration name.")
    else:
        print(f"❌ FAILURE: Schema name mismatch! Expected '{target_name}', got '{schema['name']}'")

if __name__ == "__main__":
    verify_renaming()
