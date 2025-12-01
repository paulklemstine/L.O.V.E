from typing import Dict, Any
from core.state import DeepAgentState
from core.memory.memory_folding_agent import MemoryFoldingAgent

async def fold_memory_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Node responsible for folding memory.
    Updates Episodic, Working, and Tool memory based on recent messages.
    """
    agent = MemoryFoldingAgent()
    messages = state["messages"]
    
    # We pass copies or references? Pydantic models are mutable if not frozen.
    # But state updates in LangGraph usually merge.
    
    episodic = await agent.fold_episodic(messages, state["episodic_memory"])
    working = await agent.update_working(messages, state["working_memory"])
    tool = await agent.update_tool(messages, state["tool_memory"])
    
    return {
        "episodic_memory": episodic,
        "working_memory": working,
        "tool_memory": tool
    }
