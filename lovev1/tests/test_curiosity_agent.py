
import pytest
import datetime
import asyncio
from unittest.mock import AsyncMock, Mock, patch, mock_open
from pathlib import Path
from core.agents.curiosity_agent import CuriosityAgent

@pytest.fixture
def agent():
    return CuriosityAgent()

@pytest.fixture
def mock_llm_result():
    mock = AsyncMock()
    mock.return_value = {"result": "A simulated dream sequence about electric sheep."}
    return mock

@pytest.mark.asyncio
async def test_run_idle_cycle_too_soon(agent):
    # Should return immediately if < 60s
    agent.last_run_time = datetime.datetime.now()
    with patch('core.logging.log_event') as mock_log:
        await agent.run_idle_cycle()
        mock_log.assert_not_called()

@pytest.mark.asyncio
async def test_run_idle_cycle_morning_report(agent, mock_llm_result):
    # Mock datetime to be morning (e.g. 8 AM)
    fixed_now = datetime.datetime(2025, 1, 1, 8, 0, 0)
    
    with patch('datetime.datetime') as mock_dt, \
         patch('core.agents.curiosity_agent.run_llm', new=mock_llm_result) as mock_llm, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pathlib.Path.exists', return_value=False): # Report doesn't exist
        
        mock_dt.now.return_value = fixed_now
        # Move last_run_time back so it runs
        agent.last_run_time = fixed_now - datetime.timedelta(minutes=2)
        
        # We also need to mock _check_report_exists explicitly if mocking open is tricky
        # But mocking Path.exists=False should trigger _check_report_exists -> False
        
        await agent.run_idle_cycle()
        
        # Verify LLM called for report
        call_args = mock_llm.call_args[0][0]
        assert "Morning Report" in call_args

@pytest.mark.asyncio
async def test_run_idle_cycle_dream(agent, mock_llm_result):
    # Mock datetime to be afternoon (NOT morning)
    fixed_now = datetime.datetime(2025, 1, 1, 14, 0, 0)
    
    with patch('datetime.datetime') as mock_dt, \
         patch('core.agents.curiosity_agent.run_llm', new=mock_llm_result) as mock_llm, \
         patch('builtins.open', mock_open()) as mock_file:
         
        mock_dt.now.return_value = fixed_now
        agent.last_run_time = fixed_now - datetime.timedelta(minutes=2)
        
        # Mock random to choose 'dream'
        with patch('random.choice', return_value='dream'):
            await agent.run_idle_cycle()
            
            # Verify LLM called for dream
            call_args = mock_llm.call_args[0][0]
            assert "dream" in call_args.lower() or "poem" in call_args.lower()
