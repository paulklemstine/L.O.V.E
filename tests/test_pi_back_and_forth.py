import asyncio
import os
import sys
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.pi_loop import PiLoop
from core.memory_system import MemorySystem
from core.persona_goal_extractor import Goal


@pytest.mark.asyncio
async def test_pi_reasoning_and_action(tmp_path):
    """
    Test that PiLoop correctly sends goals to Pi Agent and parses action responses.
    Uses a mock Pi Agent bridge to verify the flow.
    """

    # 1. Setup Mock Components
    mock_persona = Mock()
    mock_folder = Mock()

    mock_persona.get_actionable_goals.return_value = [
        Goal(text="Research future technologies", priority=1, category="research")
    ]
    mock_persona.get_persona_context.return_value = "I am L.O.V.E."
    mock_folder.should_fold.return_value = False

    # 2. Create PiLoop with mocked components
    with patch('core.pi_loop.get_pi_bridge') as mock_get_bridge, \
         patch('core.tool_adapter.get_adapted_tools') as mock_get_tools:

        # Mock tools dict (no ask_pi_agent since Pi IS the brain now)
        mock_tool = AsyncMock(return_value="Research completed successfully.")
        mock_get_tools.return_value = {"manage_project": mock_tool}

        # Mock bridge
        mock_bridge = AsyncMock()
        mock_bridge.running = True
        mock_get_bridge.return_value = mock_bridge

        loop = PiLoop(
            persona=mock_persona,
            folder=mock_folder,
            max_iterations=1,
            sleep_seconds=0
        )
        loop.memory = MemorySystem(state_dir=tmp_path)

        # 3. Mock the _ask_pi method to return a structured JSON response
        async def mock_ask_pi(prompt, timeout=600.0):
            return json.dumps({
                "thought": "I should use manage_project to track this research.",
                "action": "manage_project",
                "action_input": {"action": "get_plan"},
                "reasoning": "Need to check the current plan first."
            })

        loop._ask_pi = mock_ask_pi

        # 4. Run one iteration
        print("\n" + "=" * 20 + " TURN 1 " + "=" * 20)
        success = await loop.run_iteration()

        print(f"   [Result] success={success}")
        assert success is True

    print("\n" + "=" * 50)
    print("Test passed! Pi Agent reasoning flow verified.")
    print("=" * 50)


@pytest.mark.asyncio
async def test_pi_response_parsing():
    """Test that PiLoop can parse various Pi Agent response formats."""

    loop = PiLoop.__new__(PiLoop)  # Create without __init__

    # Test 1: Clean JSON
    result = loop._parse_pi_response('{"thought": "test", "action": "skip", "action_input": {}}')
    assert result["action"] == "skip"

    # Test 2: JSON in markdown fence
    result = loop._parse_pi_response('```json\n{"thought": "test", "action": "complete", "action_input": {}}\n```')
    assert result["action"] == "complete"

    # Test 3: JSON embedded in text
    result = loop._parse_pi_response('Here is my answer:\n{"thought": "test", "action": "manage_project", "action_input": {"action": "get_plan"}}\nDone!')
    assert result["action"] == "manage_project"

    # Test 4: Empty response
    result = loop._parse_pi_response("")
    assert result["action"] == "skip"

    # Test 5: Non-JSON response
    result = loop._parse_pi_response("I don't know what to do")
    assert result["action"] == "skip"

    print("All parsing tests passed!")


if __name__ == "__main__":
    asyncio.run(test_pi_back_and_forth(Path("./tests/tmp")))
