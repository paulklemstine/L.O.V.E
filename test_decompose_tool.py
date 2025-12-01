#!/usr/bin/env python3
"""
Quick test to verify the decompose_and_solve_subgoal tool is properly registered
and can be imported without errors.
"""

import sys
import asyncio

async def test_tool_registration():
    """Test that the tool is properly registered."""
    print("Testing decompose_and_solve_subgoal tool registration...")
    
    try:
        # Import the tool
        from core.tools import decompose_and_solve_subgoal, ToolRegistry
        print("✓ Successfully imported decompose_and_solve_subgoal")
        
        # Verify it's a coroutine function
        import inspect
        if inspect.iscoroutinefunction(decompose_and_solve_subgoal):
            print("✓ decompose_and_solve_subgoal is an async function")
        else:
            print("✗ ERROR: decompose_and_solve_subgoal is not an async function")
            return False
        
        # Create a tool registry and register the tool
        registry = ToolRegistry()
        registry.register_tool(
            name="decompose_and_solve_subgoal",
            tool=decompose_and_solve_subgoal,
            metadata={
                "description": "Test registration",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "sub_goal": {"type": "string"}
                    },
                    "required": ["sub_goal"]
                }
            }
        )
        print("✓ Successfully registered tool in ToolRegistry")
        
        # Verify it's in the registry
        if "decompose_and_solve_subgoal" in registry.get_tool_names():
            print("✓ Tool found in registry")
        else:
            print("✗ ERROR: Tool not found in registry")
            return False
        
        # Test error handling (no sub_goal provided)
        result = await decompose_and_solve_subgoal()
        if "Error" in result and "sub_goal" in result:
            print("✓ Error handling works correctly (missing sub_goal)")
        else:
            print(f"✗ Unexpected result for missing sub_goal: {result}")
            return False
        
        # Test error handling (no engine provided)
        result = await decompose_and_solve_subgoal(sub_goal="test goal")
        if "Error" in result and "engine" in result:
            print("✓ Error handling works correctly (missing engine)")
        else:
            print(f"✗ Unexpected result for missing engine: {result}")
            return False
        
        print("\n✅ All tests passed! The tool is properly implemented and registered.")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tool_registration())
    sys.exit(0 if success else 1)
