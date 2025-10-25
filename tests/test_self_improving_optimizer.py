import pytest
import os
from unittest.mock import patch, AsyncMock
from core.agents.self_improving_optimizer import SelfImprovingOptimizer

@pytest.mark.asyncio
async def test_self_improving_optimizer_initialization():
    """
    Validates that the SelfImprovingOptimizer can be initialized.
    """
    print("\n--- Running Test for SelfImprovingOptimizer Initialization ---")
    optimizer = SelfImprovingOptimizer()
    assert optimizer is not None
    print("--- Test Passed: SelfImprovingOptimizer Initialization is sound. ---")

@pytest.mark.asyncio
@patch('core.gemini_react_engine.execute_reasoning_task')
async def test_improve_module_runs(mock_reasoning_task):
    """
    Validates that the improve_module method runs without errors.
    """
    print("\n--- Running Test for improve_module ---")
    os.environ["GEMINI_API_KEY"] = "test_key"
    mock_reasoning_task.return_value = {
        "result": '{"thought": "I have improved the module.", "action": {"tool_name": "Finish", "arguments": {}}}'
    }

    optimizer = SelfImprovingOptimizer()
    result = await optimizer.improve_module("core/tools.py", "Improve the read_file function.")

    assert "Goal accomplished" in result
    mock_reasoning_task.assert_called_once()
    print("--- Test Passed: improve_module runs. ---")
