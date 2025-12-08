from typing import Dict, Any
from langgraph.graph import StateGraph, END
from core.state import DeepAgentState
from core.llm_api import run_llm
from langchain_core.messages import AIMessage
from core.nodes.static_analysis import static_analysis_node

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
    
    CRITICAL SECURITY INSTRUCTIONS:
    - You MUST write secure code.
    - Avoid `shell=True` in subprocess calls. Use `shlex.split()` instead.
    - Do not hardcode secrets or API keys.
    - If you receive a SECURITY CRITICAL error from the analysis tool, you MUST fix it.
    - If you believe a security warning is a false positive, you may use `# nosec` to suppress it, BUT you must provide a comment explanation.
    
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
    workflow.add_node("static_analysis", static_analysis_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("test_runner", test_runner_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "static_analysis")
    
    def route_analysis(state: DeepAgentState):
        """
        Determines next step based on analysis results.
        """
        working_memory = state.get("working_memory", {})
        status = working_memory.get("analysis_status", "failed")
        
        # Safety Check: Infinite Loop Prevention
        iterations = working_memory.get("analysis_iterations", 0)
        if iterations > 5:
            # Fallback to human review if agent loops too many times
            return "reviewer" 
            
        if status == "passed":
            return "reviewer"
        else:
            return "coder"

    workflow.add_conditional_edges(
        "static_analysis",
        route_analysis,
        {
            "coder": "coder",       # Loop back for fixes
            "reviewer": "reviewer"  # Proceed if clean
        }
    )
    
    def check_review(state: DeepAgentState):
        # Check review status
        last_msg = state["messages"][-1].content
        if "APPROVED" in last_msg:
            return "test_runner"
        return "coder"

    workflow.add_conditional_edges("reviewer", check_review)
    workflow.add_edge("test_runner", END)
    
    return workflow.compile()
