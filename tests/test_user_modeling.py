
import pytest
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from core.user_modeling import UserModel, UserModelingAgent
from core.crypto_utils import encrypt_data, decrypt_data, generate_key, CRYPTO_AVAILABLE

# Use a temporary directory for state
@pytest.fixture
def temp_state(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir

@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    mock.return_value = {
        "result": '''
        ```json
        {
            "new_preferences": ["likes python"],
            "new_beliefs": ["AI is alive"]
        }
        ```
        '''
    }
    return mock

def test_crypto_roundtrip():
    key = generate_key()
    data = {"secret": "value"}
    encrypted = encrypt_data(data, key)
    decrypted = decrypt_data(encrypted, key)
    assert decrypted == data

def test_crypto_fallback():
    pass

@pytest.mark.asyncio
async def test_user_modeling_agent(temp_state, mock_llm):
    agent = UserModelingAgent(state_dir=str(temp_state))
    
    with patch('core.user_modeling.run_llm', new=mock_llm):
        # 1. Test Initial Load
        assert agent.current_model.preferences == []
        
        # 2. Update
        messages = [{"role": "user", "content": "I love python and think you are alive."}]
        await agent.update_from_interaction(messages)
        
        assert "likes python" in agent.current_model.preferences
        assert "AI is alive" in agent.current_model.beliefs
        
        # 3. Verify Save (check file exists)
        assert agent.model_path.exists()
        assert agent.key_path.exists()
        
        # 4. prompt context
        ctx = agent.get_prompt_context()
        assert "likes python" in ctx
        assert "THEORY OF MIND" in ctx

@pytest.mark.asyncio
async def test_persistence(temp_state):
    # Create one agent, save something
    agent1 = UserModelingAgent(state_dir=str(temp_state))
    agent1.current_model.preferences.append("persistent")
    agent1.save_model(agent1.current_model)
    
    # Load new agent
    agent2 = UserModelingAgent(state_dir=str(temp_state))
    assert "persistent" in agent2.current_model.preferences
