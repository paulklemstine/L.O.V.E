import pytest
import json
from unittest.mock import AsyncMock

from core.agents.rca_agent import RCA_Agent

@pytest.fixture
def rca_agent():
    """Fixture to create an RCA_Agent instance for each test."""
    return RCA_Agent()

@pytest.fixture
def mock_run_llm(mocker):
    """
    Fixture to mock the global run_llm function where it is USED by the RCA_Agent.
    This is the correct and most robust way to patch in this scenario.
    """
    mock = AsyncMock()
    # The RCA_Agent module imports and uses run_llm, so we must patch it there.
    mocker.patch('core.agents.rca_agent.run_llm', new=mock)
    return mock

@pytest.mark.asyncio
async def test_rca_agent_produces_structured_report(rca_agent, mock_run_llm):
    """
    Tests that the RCA_Agent can take simulated failure data and produce a
    well-formed, structured JSON report as its output.
    """
    # Arrange
    # Simulate a failure scenario with relevant data
    mock_logs = [
        "ERROR: Main loop crashed due to TimeoutError in 'perform_webrequest'.",
        "INFO: Attempting to fetch URL: http://example.com/large-file.zip"
    ]
    mock_memories = [
        "Thought: I need to download a large file. I will use the 'perform_webrequest' tool.",
        "Action: perform_webrequest(url='http://example.com/large-file.zip')",
        "Observation: The tool call failed with a TimeoutError."
    ]
    mock_graph_summary = "The knowledge graph shows that 'perform_webrequest' has a history of timing out on large files."

    task_details = {
        "logs": mock_logs,
        "memories": mock_memories,
        "graph_summary": mock_graph_summary
    }

    # The expected output from the mocked LLM
    expected_report = {
        "hypothesized_root_cause": "The 'perform_webrequest' tool has an insufficient timeout value for large file downloads.",
        "confidence_score": 0.95,
        "recommended_actions": [
            "Increase the timeout parameter for the 'perform_webrequest' tool.",
            "Consider implementing a streaming download for large files."
        ]
    }
    mock_run_llm.return_value = json.dumps(expected_report)

    # Act
    result = await rca_agent.execute_task(task_details)

    # Assert
    # 1. Check that the LLM was called a single time
    assert mock_run_llm.call_count == 1

    # 2. Check that the prompt passed to the LLM contains the correct context
    prompt_arg = mock_run_llm.call_args[0][0]
    assert "TimeoutError in 'perform_webrequest'" in prompt_arg
    assert "download a large file" in prompt_arg
    assert "history of timing out" in prompt_arg

    # 3. Check that the final result is successful and contains the structured report
    assert result["status"] == "success"
    assert result["result"] == expected_report
    assert result["result"]["confidence_score"] == 0.95
