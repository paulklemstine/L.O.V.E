"""
Manual Verification for Evolutionary Agent
"""

import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.getcwd())

# Mock ToolRegistry to avoid loading real tools
import core.agents.evolutionary_agent
core.agents.evolutionary_agent.get_global_registry = MagicMock(return_value=MagicMock())

from core.agents.evolutionary_agent import EvolutionaryAgent
from core.evolution_state import EvolutionarySpecification
from core.tool_validator import ValidationResult

async def verify():
    print("Starting verification...")
    
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="def fixed(): pass")
    
    # Create Agent
    try:
        agent = EvolutionaryAgent(llm_client=mock_llm)
        print("Agent initialized.")
    except Exception as e:
        print(f"FAILED to init agent: {e}")
        return

    # Mock Fabricator logic
    agent.fabricator.fabricate_tool = AsyncMock(return_value={
        "success": True,
        "file_path": "dummy_path.py",
        "code": "dummy code"
    })
    
    # Mock Validator: Fail first, then pass
    agent.validator.validate = AsyncMock(side_effect=[
        ValidationResult(passed=False, error_message="Syntax Error"),
        ValidationResult(passed=True)
    ])
    
    # Mock Finalize (to avoid file ops)
    agent._finalize_tool = AsyncMock(return_value=True)
    
    # Mock Refine
    agent._refine_code = AsyncMock(return_value="fixed code")
    
    # Create Spec
    spec = EvolutionarySpecification(
        functional_name="test_tool",
        required_arguments={},
        expected_output="str",
        id="test-123"
    )
    
    # Mock writing file - we will use a temp file instead

        
    # We'll monkeypatch open on the module if possible, or just catch the error
    # Actually, agent._process_single_spec calls open().
    # Let's mock open globally for this script if we can, or just mock `_refine_code` to verify the loop flow
    # and skip the write part by mocking `open` context manager?
    
    # Let's simple-mock the write in the agent method? No, better to patch.
    # We'll rely on the logic:
    # It calls validate -> fail
    # Calls refine -> success
    # Calls validate -> success
    # Calls finalize -> success
    
    # We need to mock open() to avoid IO error on 'dummy_path.py'
    # Actually, we can just let it write to a temp file if we give a real path
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
        real_path = tmp.name
    
    # Initial write by fabricator is assumed done? 
    # fabricator returns path. Agent reads/writes to it.
    
    # Update mock to return real path
    agent.fabricator.fabricate_tool = AsyncMock(return_value={
        "success": True,
        "file_path": real_path,
        "code": "original code"
    })
    
    print(f"Running process_single_spec with path {real_path}...")
    result = await agent._process_single_spec(spec)
    
    print(f"Result: {result}")
    print(f"Validate calls: {agent.validator.validate.call_count}")
    print(f"Refine called: {agent._refine_code.called}")
    
    if result and agent.validator.validate.call_count == 2 and agent._refine_code.called:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
        
    # Cleanup
    if os.path.exists(real_path):
        os.remove(real_path)

if __name__ == "__main__":
    asyncio.run(verify())
