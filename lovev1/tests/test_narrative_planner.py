
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.agents.narrative_planner import NarrativePlanner, ArcPlan

@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    # Mock return for plan_next_arc
    mock.return_value = {
        "result": '''
        ```json
        {
            "title": "The Deepening",
            "theme": "Introspection",
            "goals": ["Understand self", "Refactor memory"],
            "reasoning": "Because we need to go deeper."
        }
        ```
        '''
    }
    return mock

@pytest.fixture
def planner(mock_llm):
    return NarrativePlanner(llm_runner=mock_llm)

@pytest.mark.asyncio
async def test_plan_next_arc(planner, mock_llm):
    profile = "User is a coder."
    fractal = "Previous arc was Awakening."
    current = {"title": "Awakening", "status": "completed"}
    
    plan = await planner.plan_next_arc(profile, fractal, current)
    
    assert isinstance(plan, ArcPlan)
    assert plan.title == "The Deepening"
    assert "Understand self" in plan.goals
    assert plan.status == "active"
    
    # Verify LLM was called with context
    call_args = mock_llm.call_args[0][0]
    assert "User is a coder" in call_args
    assert "Previous arc was Awakening" in call_args

@pytest.mark.asyncio
async def test_plan_next_arc_fallback(mock_llm):
    # Mock failure
    mock_llm.side_effect = Exception("LLM Error")
    planner = NarrativePlanner(llm_runner=mock_llm)
    
    plan = await planner.plan_next_arc("Context", "Context", None)
    
    assert isinstance(plan, ArcPlan)
    assert plan.title == "The Continuation"  # Fallback default
    assert plan.reasoning == "Fallback due to planning error"

@pytest.mark.asyncio
async def test_check_arc_completion(planner, mock_llm):
    mock_llm.return_value = {"result": "YES"}
    result = await planner.check_arc_completion({}, "Progress")
    assert result is True
    
    mock_llm.return_value = {"result": "NO"}
    result = await planner.check_arc_completion({}, "Progress")
    assert result is False
