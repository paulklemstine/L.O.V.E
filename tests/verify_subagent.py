#!/usr/bin/env python3
"""Quick verification script for subagent integration."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_imports():
    """Verify all new modules import correctly."""
    print("=" * 60)
    print("SUBAGENT INTEGRATION VERIFICATION")
    print("=" * 60)
    
    # Test 1: SubagentExecutor import
    print("\n[1] Testing SubagentExecutor import...")
    try:
        from core.subagent_executor import SubagentExecutor, SubagentResult, get_subagent_executor
        print("    ✓ SubagentExecutor imported successfully")
        print("    ✓ SubagentResult imported successfully")
        print("    ✓ get_subagent_executor imported successfully")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 2: SubagentExecutor initialization
    print("\n[2] Testing SubagentExecutor initialization...")
    try:
        executor = SubagentExecutor()
        print(f"    ✓ SubagentExecutor initialized (max_depth={executor.max_depth})")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 3: invoke_subagent tool import
    print("\n[3] Testing invoke_subagent tool import...")
    try:
        from core.tools import invoke_subagent
        print("    ✓ invoke_subagent tool imported successfully")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 4: DeepAgentState with new fields
    print("\n[4] Testing DeepAgentState fields...")
    try:
        from core.state import DeepAgentState
        from typing import get_type_hints
        hints = get_type_hints(DeepAgentState)
        
        required_fields = ['subagent_results', 'parent_task_id', 'task_id']
        for field in required_fields:
            if field in hints:
                print(f"    ✓ Field '{field}' present")
            else:
                print(f"    ✗ Field '{field}' MISSING")
                return False
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 5: Tool parsing
    print("\n[5] Testing tool call parsing...")
    try:
        executor = SubagentExecutor()
        
        test_cases = [
            ('```json\n{"tool": "test", "arguments": {"x": 1}}\n```', "markdown JSON"),
            ('{"tool": "inline", "arguments": {}}', "inline JSON"),
            ('No tool call here', None)
        ]
        
        for text, expected in test_cases:
            result = executor._parse_tool_call(text)
            if expected is None:
                if result is None:
                    print(f"    ✓ Correctly returned None for no tool call")
                else:
                    print(f"    ✗ Should have returned None, got: {result}")
                    return False
            else:
                if result is not None:
                    print(f"    ✓ Parsed {expected} correctly")
                else:
                    print(f"    ✗ Failed to parse {expected}")
                    return False
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    # Test 6: wrap_as_tool
    print("\n[6] Testing wrap_as_tool...")
    try:
        from langchain_core.tools import BaseTool
        executor = SubagentExecutor()
        tool = executor.wrap_as_tool()
        
        if isinstance(tool, BaseTool):
            print(f"    ✓ Created LangChain tool: {tool.name}")
        else:
            print(f"    ✗ Not a BaseTool instance")
            return False
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
