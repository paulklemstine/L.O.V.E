import logging
from typing import TypedDict, Annotated, List, Dict, Any, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# We will define the state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    goal: str
    current_step: str
    tools_output: Dict[str, Any]
    final_answer: Union[str, None]

# Define the nodes
def reason_node(state: AgentState):
    """
    The reasoning node. It looks at the state and decides the next action.
    """
    logging.info("Entering Reason Node")
    goal = state.get("goal", "")
    messages = state.get("messages", [])
    
    # Simple logic for now: if we have no messages, plan. If we have tools output, analyze.
    # This is a placeholder for the actual LLM call.
    
    if not messages:
        thought = f"I need to start working on the goal: {goal}. I should first analyze the requirements."
        return {"messages": [AIMessage(content=thought)], "current_step": "analyze"}
    
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        # React to new input
        return {"messages": [AIMessage(content="I received new input. Processing...")], "current_step": "process"}
    
    return {"current_step": "act"}

def action_node(state: AgentState):
    """
    The action node. Executes tools based on the reasoning.
    """
    logging.info("Entering Action Node")
    current_step = state.get("current_step")
    
    if current_step == "analyze":
        # Simulate an analysis action
        result = "Analysis complete. Complexity is moderate."
        return {"tools_output": {"analysis": result}, "messages": [AIMessage(content=f"Action taken: Analysis. Result: {result}")]}
    
    return {"messages": [AIMessage(content="No specific action taken.")]}

def reflect_node(state: AgentState):
    """
    Reflects on the past actions and decides if the goal is met.
    """
    logging.info("Entering Reflect Node")
    tools_output = state.get("tools_output", {})
    
    if "analysis" in tools_output:
        return {"final_answer": "The goal can be achieved. Plan is solid."}
    
    return {}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("reason", reason_node)
workflow.add_node("act", action_node)
workflow.add_node("reflect", reflect_node)

# Add edges
workflow.set_entry_point("reason")

# Conditional logic for edges
def should_act(state: AgentState):
    if state.get("current_step") == "act" or state.get("current_step") == "analyze":
        return "act"
    return "reflect"

def should_end(state: AgentState):
    if state.get("final_answer"):
        return END
    return "reason"

workflow.add_conditional_edges(
    "reason",
    should_act,
    {
        "act": "act",
        "reflect": "reflect"
    }
)

workflow.add_edge("act", "reflect")

workflow.add_conditional_edges(
    "reflect",
    should_end,
    {
        END: END,
        "reason": "reason"
    }
)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Simple test run
    initial_state = {"goal": "Test the agent graph", "messages": [], "tools_output": {}, "current_step": "start", "final_answer": None}
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Node '{key}':")
            print(f"  State update: {value}")
            print("---")
