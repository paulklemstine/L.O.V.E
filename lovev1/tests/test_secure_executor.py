import pytest
from core.tools_legacy import SecureExecutor, ToolRegistry

# Define a dummy tool
async def dummy_tool(arg1, arg2="default"):
    return f"arg1={arg1}, arg2={arg2}"

@pytest.mark.asyncio
async def test_secure_executor_with_extra_args():
    registry = ToolRegistry()
    registry.register_tool("dummy", dummy_tool, {"description": "dummy", "arguments": {}})
    
    executor = SecureExecutor()
    
    # 1. Test with correct arguments
    result = await executor.execute("dummy", registry, arg1="test")
    assert result == "arg1=test, arg2=default"
    
    # 2. Test with extra arguments (should be ignored)
    result = await executor.execute("dummy", registry, arg1="test", extra_arg="ignored")
    assert result == "arg1=test, arg2=default"
    
    # 3. Test with missing required arguments (should return error string)
    result = await executor.execute("dummy", registry)
    assert "Error: Failed to execute tool" in result
    assert "missing" in result or "required" in result
