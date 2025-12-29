#!/usr/bin/env python3
"""
Quick verification test for Story 1.1-1.4 changes.
"""
import sys
sys.path.insert(0, '/home/raver1975/L.O.V.E')

def test_imports():
    """Test that new modules import correctly."""
    try:
        from core.legacy_compat import ToolRegistry
        print("✅ legacy_compat.ToolRegistry imports OK")
    except Exception as e:
        print(f"❌ legacy_compat.ToolRegistry failed: {e}")
        return False
    
    try:
        from core.secure_executor import SecureExecutor
        print("✅ secure_executor.SecureExecutor imports OK")
    except Exception as e:
        print(f"❌ secure_executor.SecureExecutor failed: {e}")
        return False
    
    try:
        from core.tool_registry import ToolRegistry as NewRegistry, tool_schema, get_global_registry
        print("✅ tool_registry imports OK")
    except Exception as e:
        print(f"❌ tool_registry failed: {e}")
        return False
        
    return True

def test_registry_functions():
    """Test registry functionality."""
    try:
        from core.tool_registry import ToolRegistry, tool_schema
        
        reg = ToolRegistry()
        
        @tool_schema
        def test_tool(query: str) -> str:
            """A test tool.
            
            Args:
                query: Search query
            """
            return f"Result for {query}"
        
        reg.register(test_tool)
        schemas = reg.get_all_tool_schemas()
        
        assert len(schemas) == 1, f"Expected 1 schema, got {len(schemas)}"
        assert schemas[0]["name"] == "test_tool"
        print("✅ get_all_tool_schemas() works correctly")
        
        return True
    except Exception as e:
        print(f"❌ Registry functions failed: {e}")
        return False

def test_legacy_compat():
    """Test legacy compatibility layer."""
    try:
        from core.legacy_compat import LegacyToolRegistry
        
        reg = LegacyToolRegistry()
        
        def old_tool(x):
            """An old tool."""
            return x
        
        reg.register_tool("old_tool", old_tool, {
            "description": "An old tool",
            "arguments": {"type": "object", "properties": {}}
        })
        
        tool = reg.get_tool("old_tool")
        assert tool is not None
        print("✅ LegacyToolRegistry.register_tool() works")
        
        names = reg.get_tool_names()
        assert "old_tool" in names
        print("✅ LegacyToolRegistry.get_tool_names() works")
        
        return True
    except Exception as e:
        print(f"❌ Legacy compat failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Tool Registry Enhancement - Verification Tests")
    print("=" * 50)
    
    all_passed = True
    
    print("\n--- Testing Imports ---")
    all_passed &= test_imports()
    
    print("\n--- Testing Registry Functions ---")
    all_passed &= test_registry_functions()
    
    print("\n--- Testing Legacy Compatibility ---")
    all_passed &= test_legacy_compat()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All verification tests PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED")
        sys.exit(1)
