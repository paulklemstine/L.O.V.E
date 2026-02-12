import asyncio
import os
import sys
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.deep_loop import DeepLoop
from core.memory_system import MemorySystem
from core.persona_goal_extractor import Goal

@pytest.mark.asyncio
async def test_pi_back_and_forth(tmp_path):
    """
    Test the back-and-forth communication between L.O.V.E. and Pi Agent.
    """
    
    # 1. Setup Mock Components
    mock_llm = AsyncMock()
    mock_persona = Mock()
    mock_folder = Mock()
    
    mock_persona.get_actionable_goals.return_value = [
        Goal(text="Research future technologies", priority=1, category="research")
    ]
    mock_persona.get_persona_context.return_value = "I am L.O.V.E."
    mock_folder.should_fold.return_value = False
    
    # Mock ask_pi_agent tool
    mock_pi_tool = AsyncMock()
    
    async def ask_pi_agent_wrapper(prompt: str, timeout: float = 600.0) -> str:
        """Send a prompt to the Pi Agent and get a response."""
        print(f"   [Tool Call] ask_pi_agent(prompt='{prompt}')")
        res = await mock_pi_tool(prompt, timeout)
        print(f"   [Pi Response] {res}")
        return res
    
    # 2. Define LLM and Pi behaviors for Turn 1
    mock_llm.generate_json_async.side_effect = [
        # Turn 1 Decision
        {
            "thought": "I should ask Pi for a research outline.",
            "action": "ask_pi_agent",
            "action_input": {"prompt": "Give me a research outline for future tech."},
            "reasoning": "Need a starting point."
        },
        # Turn 2 Decision (after seeing Pi's response)
        {
            "thought": "Pi asked for a focus. I will choose AI Ethics.",
            "action": "ask_pi_agent",
            "action_input": {"prompt": "Focus on AI Ethics, please."},
            "reasoning": "Answering Pi's question to continue the research."
        }
    ]
    
    # Pi's response for Turn 1
    mock_pi_tool.side_effect = [
        "Outline: 1. AI, 2. BioTech. Which one should I focus on first?",
        "Focused Research: AI Ethics is a critical field..."
    ]
    
    # 3. Initialize DeepLoop with patched tools
    with patch('core.tool_adapter.get_adapted_tools') as mock_get_tools:
        mock_get_tools.return_value = {"ask_pi_agent": ask_pi_agent_wrapper}
        
        loop = DeepLoop(
            llm=mock_llm,
            persona=mock_persona,
            folder=mock_folder
        )
        loop.memory = MemorySystem(state_dir=tmp_path)
        
        # 4. Run Turn 1
        print("\n" + "="*20 + " TURN 1 " + "="*20)
        success1 = await loop.run_iteration()
        
        # Print LLM decision for Turn 1
        turn1_decision = mock_llm.generate_json_async.call_args_list[0][1]
        print(f"   [LLM Thought] I should ask Pi for a research outline.")
        print(f"   [LLM Action] ask_pi_agent")
        
        assert success1 is True
        assert loop.last_pi_interaction["prompt"] == "Give me a research outline for future tech."
        assert "AI, 2. BioTech" in str(loop.last_pi_interaction["response"])
        
        # 5. Run Turn 2
        print("\n" + "="*20 + " TURN 2 " + "="*20)
        
        # Before iteration, check the context that will be passed
        pi_interaction = (
            f"Your last prompt to Pi: {loop.last_pi_interaction['prompt']}\n"
            f"Pi's Response: {loop.last_pi_interaction['response']}"
        )
        print(f"   [Injected Context]\n{pi_interaction}")
        
        success2 = await loop.run_iteration()
        
        # Print LLM decision for Turn 2
        print(f"   [LLM Thought] Pi asked for a focus. I will choose AI Ethics.")
        print(f"   [LLM Action] ask_pi_agent")

        assert success2 is True
        
        # Verify Turn 2 outcome
        assert loop.last_pi_interaction["prompt"] == "Focus on AI Ethics, please."
        assert "AI Ethics is a critical field" in str(loop.last_pi_interaction["response"])
        
        # 6. Verify LLM prompt context for Turn 2
        args, kwargs = mock_llm.generate_json_async.call_args_list[1]
        user_prompt = kwargs.get('prompt')
        assert "Your last prompt to Pi: Give me a research outline" in user_prompt
        assert "Pi's Response: Outline: 1. AI, 2. BioTech" in user_prompt
    
    print("\n" + "="*50)
    print("Test passed! Back-and-forth communication verified.")
    print("="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pi_back_and_forth(Path("./tests/tmp")))
