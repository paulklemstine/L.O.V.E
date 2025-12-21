from typing import Dict, Any, AsyncGenerator
from core.graph import create_deep_agent_graph
from core.state import DeepAgentState
from core.memory.schemas import EpisodicMemory, WorkingMemory, ToolMemory
from langchain_core.messages import HumanMessage, BaseMessage

class DeepAgentRunner:
    def __init__(self):
        self.graph = create_deep_agent_graph()
        # Initialize state with empty memories
        self.state: DeepAgentState = {
            "messages": [],
            "episodic_memory": EpisodicMemory(),
            "working_memory": WorkingMemory(),
            "tool_memory": ToolMemory(),
            "next_node": None,
            "recursion_depth": 0,
            "stop_reason": None,
            "tool_query": None,
            "retrieved_tools": []
        }

    async def run(self, user_input: str, mandate: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the agent with the given input.
        Yields updates from the graph execution.
        """
        # Add user message
        self.state["messages"].append(HumanMessage(content=user_input))
        
        # Inject mandate if provided
        if mandate:
            self.state["creator_mandate"] = mandate
        
        # Run the graph
        # astream returns a stream of outputs from nodes
        async for output in self.graph.astream(self.state):
            for node_name, node_update in output.items():
                # Update our local state tracking (optional, as graph maintains it)
                # But we might want to inspect it
                yield {node_name: node_update}
                
                # Update messages in our local state if returned
                if "messages" in node_update:
                    # Append new messages
                    # Note: LangGraph's add_messages reducer handles this in the graph state
                    # We just need to keep our local mirror in sync if we want to use it between runs
                    # But actually, we should probably pass the graph's final state back
                    pass

    def get_state(self) -> DeepAgentState:
        return self.state
