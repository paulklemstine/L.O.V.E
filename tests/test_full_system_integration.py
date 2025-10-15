import pytest
import json
from unittest.mock import patch, AsyncMock

# Add necessary imports from the 'core' directory
from core.agents.orchestrator import Orchestrator
from core.metacognition import HypothesisFormatter, ExperimentPlanner
from core.agents.analyst_agent import AnalystAgent

def test_phase_1_cognitive_architecture_initialization():
    """
    Validates that the Orchestrator and its core components are
    initialized correctly.
    """
    print("\n--- Running Test for Phase 1: Cognitive Architecture ---")

    orchestrator = Orchestrator()

    assert orchestrator is not None
    assert orchestrator.planner is not None
    assert orchestrator.tool_registry is not None
    assert orchestrator.executor is not None
    assert orchestrator.execution_engine is not None

    assert orchestrator.tool_registry.get_tool("web_search") is not None
    assert orchestrator.tool_registry.get_tool("read_file") is not None

    print("--- Phase 1 Test Passed: Cognitive Architecture is sound. ---")

@pytest.mark.asyncio
@patch('core.planning.mock_llm_call')
async def test_phase_2_action_and_planning_engine(mock_llm_call_func):
    """
    Validates the asynchronous action and planning engine.
    """
    print("\n--- Running Test for Phase 2: Action & Planning Engine ---")

    test_goal = "Summarize the latest advancements in AI"
    mock_plan = [
        {"step": 1, "task": "Execute web searches using the 'web_search' tool."},
        {"step": 2, "task": "Read the content using the 'read_file' tool."},
        {"step": 3, "task": "Produce a final summary report."}
    ]
    mock_llm_call_func.return_value = json.dumps(mock_plan)

    mock_article_content = "The latest advancements in AI are groundbreaking."

    with patch('core.tools.web_search', new_callable=AsyncMock) as mock_search, \
         patch('core.tools.read_file', new_callable=AsyncMock) as mock_read:

        mock_search.return_value = "[Article 1: AI Today]"
        mock_read.return_value = mock_article_content

        orchestrator = Orchestrator()

        orchestrator.tool_registry.register_tool("web_search", mock_search)
        orchestrator.tool_registry.register_tool("read_file", mock_read)

        result = await orchestrator.execute_goal(test_goal)

        assert result.get('status') == 'Success'
        assert result.get('final_result') == mock_article_content
        mock_llm_call_func.assert_called_once()
        mock_search.assert_awaited_once()
        mock_read.assert_awaited_once()

    print("--- Phase 2 Test Passed: Action & Planning Engine is functional. ---")

@pytest.mark.asyncio
async def test_phase_3_metacognitive_evolution_loop():
    """
    Validates the asynchronous metacognitive evolution loop.
    """
    print("\n--- Running Test for Phase 3: Metacognitive Evolution Loop ---")

    mock_logs = [
        {"event": "tool_start", "tool_name": "web_search", "token_usage": 2100},
        {"event": "tool_success", "tool_name": "web_search", "result": "Full HTML page"}
    ]

    insight = AnalystAgent.analyze_logs(mock_logs)
    assert "Insight: The web_search tool is inefficient" in insight

    hypothesis = HypothesisFormatter.format_hypothesis(insight)
    assert hypothesis.startswith("IF we")

    experiment = ExperimentPlanner.design_experiment(hypothesis)
    assert experiment['metric'] == 'token_usage_metric'

    print("--- Phase 3 Test Passed: Metacognitive Loop is functional. ---")