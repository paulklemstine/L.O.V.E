from typing import Dict, Any
from langgraph.graph import StateGraph, END
from core.state import DeepAgentState
from core.llm_api import run_llm
from langchain_core.messages import AIMessage

# --- Nodes ---

async def planner_node(state: DeepAgentState) -> Dict[str, Any]:
    messages = state["messages"]
    prompt = f"""
    You are the Lead Planner for the Coding Team.
    Analyze the request and create a step-by-step implementation plan.
    
    Request: {messages[-1].content}
    
    Return the plan.
    """
    response = await run_llm(prompt, purpose="planning")
    return {"messages": [AIMessage(content=f"Plan: {response.get('result')}")], "working_memory": {"current_subgoal": "coding_plan_created"}}

async def coder_node(state: DeepAgentState) -> Dict[str, Any]:
    messages = state["messages"]
    # Extract plan from history or working memory
    prompt = f"""
    You are the Senior Developer.
    Write the code based on the plan.
    
    Context: {messages[-1].content}
    
    Return the code.
    """
    response = await run_llm(prompt, purpose="coding", is_source_code=True)
    return {"messages": [AIMessage(content=response.get("result"))]}

async def reviewer_node(state: DeepAgentState) -> Dict[str, Any]:
    messages = state["messages"]
    code = messages[-1].content
    prompt = f"""
    You are the Code Reviewer.
    Review the following code for errors, bugs, and best practices.
    
    Code:
    {code}
    
    If approved, say "APPROVED". Otherwise, list the issues.
    """
    response = await run_llm(prompt, purpose="review")
    result = response.get("result", "")
    approved = "APPROVED" in result
    
    return {"messages": [AIMessage(content=result)], "review_approved": approved}

async def test_runner_node(state: DeepAgentState) -> Dict[str, Any]:
    # In a real scenario, this would run tests.
    # For now, we simulate it or use the 'execute' tool if we were binding tools.
    return {"messages": [AIMessage(content="Tests passed (simulated).")]}

# --- Graph Definition ---

def create_coding_graph():
    workflow = StateGraph(DeepAgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("test_runner", test_runner_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "reviewer")
    
    def check_review(state: DeepAgentState):
        # We need to store the approval status in the state somehow.
        # Since DeepAgentState is TypedDict, we can't easily add arbitrary keys unless defined.
        # We'll assume 'review_approved' is added to DeepAgentState or we parse the last message.
        # For this prototype, let's assume we added it to the TypedDict or use a specific field.
        # I'll check DeepAgentState definition again.
        # It doesn't have 'review_approved'. I should add it or use 'working_memory'.
        
        # For now, let's check the last message content.
        last_msg = state["messages"][-1].content
        if "APPROVED" in last_msg:
            return "test_runner"
        return "coder"

    workflow.add_conditional_edges("reviewer", check_review)
    workflow.add_edge("test_runner", END)
    
    return workflow.compile()
