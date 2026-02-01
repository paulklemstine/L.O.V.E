
import pytest
import asyncio
from core.codeact_engine import CodeActEngine

@pytest.mark.asyncio
async def test_single_line_execution():
    engine = CodeActEngine()
    result = await engine.execute("print('hello world')")
    assert result.success
    assert "hello world" in result.stdout

@pytest.mark.asyncio
async def test_multi_line_indentation():
    engine = CodeActEngine()
    code = """
x = 10
y = 20
print(f"Sum: {x + y}")
"""
    result = await engine.execute(code.strip())
    assert result.success
    assert "Sum: 30" in result.stdout

@pytest.mark.asyncio
async def test_persistent_state():
    engine = CodeActEngine()
    
    # Define a function
    await engine.execute("""
def add(a, b):
    return a + b
""")
    
    # Use the function in a later call
    result = await engine.execute("print(add(5, 7))")
    assert result.success
    assert "12" in result.stdout.strip()

@pytest.mark.asyncio
async def test_error_handling():
    engine = CodeActEngine()
    result = await engine.execute("raise ValueError('test error')")
    assert not result.success
    assert "ValueError" in result.error_type
    assert "test error" in result.stderr

@pytest.mark.asyncio
async def test_security_check():
    engine = CodeActEngine()
    result = await engine.execute("import os; os.system('ls')")
    assert not result.success
    assert result.error_type == "SecurityError"
