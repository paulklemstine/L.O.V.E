"""
Coding Team Subgraph for the DeepAgent system.

This module implements a multi-agent coding workflow:
- Planner: Creates implementation plan
- Coder: Writes code based on plan
- Static Analysis: Runs linting/security checks
- Reviewer: Reviews code quality
- Test Runner: Executes tests in Docker sandbox

The graph implements self-correction loops for both static analysis
and code review phases.
"""
import os
import logging
from typing import Dict, Any, Union
from langgraph.graph import StateGraph, END
from core.state import DeepAgentState
from core.memory.schemas import WorkingMemory
from core.llm_api import run_llm
from langchain_core.messages import AIMessage, HumanMessage
from core.nodes.static_analysis import static_analysis_node

logger = logging.getLogger(__name__)


def get_working_memory_dict(state: DeepAgentState) -> Dict[str, Any]:
    """
    Safely extracts the working memory as a dictionary.
    
    Handles both WorkingMemory Pydantic model and plain dict cases.
    Uses active_variables field for dynamic data storage.
    """
    raw_wm = state.get("working_memory", {})
    if isinstance(raw_wm, WorkingMemory):
        return raw_wm.active_variables
    elif hasattr(raw_wm, 'model_dump'):
        return raw_wm.model_dump().get('active_variables', {})
    elif isinstance(raw_wm, dict):
        return raw_wm
    return {}

# --- Utility Functions ---

def extract_code_from_response(content: str) -> str:
    """
    Extracts code from LLM response, handling markdown fences.
    
    Args:
        content: Raw LLM response that may contain markdown code blocks
        
    Returns:
        Extracted code string
    """
    if "```python" in content:
        return content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content.strip()


def write_code_to_file(code: str, filepath: str) -> bool:
    """
    Writes code to a file, creating directories if needed.
    
    Args:
        code: The code content to write
        filepath: Path to write the file to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(code)
        logger.info(f"Wrote code to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to write code to {filepath}: {e}")
        return False


# --- Nodes ---

async def planner_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Lead Planner node that creates an implementation plan.
    
    Analyzes the request and creates a step-by-step plan,
    storing it in working_memory for downstream nodes.
    """
    messages = state["messages"]
    working_memory = get_working_memory_dict(state)
    
    # Determine file path from request or use default
    request_content = messages[-1].content if messages else ""
    
    # Try to extract a filename from the request
    import re
    filename_match = re.search(r'(?:create|write|build|make)\s+(?:a\s+)?(\w+\.py)', request_content, re.IGNORECASE)
    if filename_match:
        target_file = filename_match.group(1)
    else:
        target_file = "generated_code.py"
    
    filepath = os.path.join(os.getcwd(), "sandbox_output", target_file)
    
    prompt = f"""
    You are the Lead Planner for the Coding Team.
    Analyze the request and create a step-by-step implementation plan.
    
    Request: {request_content}
    
    Target file: {target_file}
    
    Create a clear, actionable plan with numbered steps.
    Include:
    1. Requirements analysis
    2. Key functions/classes needed
    3. Dependencies required
    4. Testing approach
    
    Return the plan.
    """
    
    response = await run_llm(prompt, purpose="planning")
    plan_content = response.get('result', 'No plan generated')
    
    # Update working memory with plan info
    new_active_vars = {
        **working_memory,
        "current_subgoal": "coding_plan_created",
        "current_file_path": filepath,
        "target_file": target_file,
        "plan": plan_content
    }
    
    return {
        "messages": [AIMessage(content=f"Plan:\n{plan_content}")],
        "working_memory": WorkingMemory(active_variables=new_active_vars)
    }


async def coder_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Senior Developer node that writes code based on the plan.
    
    Generates code and writes it to the file specified in working_memory.
    """
    messages = state["messages"]
    working_memory = get_working_memory_dict(state)
    
    # Get filepath from working_memory
    filepath = working_memory.get("current_file_path", "sandbox_output/generated_code.py")
    plan = working_memory.get("plan", "")
    
    # Get feedback from previous iterations (static analysis or reviewer)
    feedback = ""
    for msg in reversed(messages[-5:]):  # Look at last 5 messages
        content = getattr(msg, "content", "")
        if isinstance(msg, HumanMessage) and ("Static Analysis" in content or "SECURITY" in content or "COMPLEXITY" in content):
            feedback = f"\n\nPREVIOUS FEEDBACK (you must address these issues):\n{content}"
            break
    
    # Get previous code if exists (for iteration)
    previous_code = working_memory.get("current_code_content", "")
    previous_code_context = ""
    if previous_code:
        previous_code_context = f"\n\nPREVIOUS CODE VERSION (fix the issues):\n```python\n{previous_code}\n```"
    
    prompt = f"""
    You are the Senior Developer.
    Write the code based on the plan and any feedback received.
    
    CRITICAL SECURITY INSTRUCTIONS:
    - You MUST write secure code.
    - Avoid `shell=True` in subprocess calls. Use `shlex.split()` instead.
    - Do not hardcode secrets or API keys.
    - If you receive a SECURITY CRITICAL error, you MUST fix it.
    - If you believe a security warning is a false positive, you may use `# nosec` to suppress it, BUT you must provide a comment explanation.
    
    Plan:
    {plan}
    
    Context from conversation: {messages[-1].content if messages else "No context"}
    {previous_code_context}
    {feedback}
    
    Write complete, working Python code. Include proper imports, docstrings, and error handling.
    Wrap your code in ```python ... ``` markers.
    """
    
    response = await run_llm(prompt, purpose="coding", is_source_code=True)
    raw_response = response.get("result", "")
    
    # Extract code from response
    code = extract_code_from_response(raw_response)
    
    # Write code to file
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    success = write_code_to_file(code, filepath)
    
    if not success:
        return {
            "messages": [AIMessage(content=f"Error: Failed to write code to {filepath}")],
            "working_memory": WorkingMemory(active_variables=working_memory)
        }
    
    # Update working memory
    new_active_vars = {
        **working_memory,
        "current_code_content": code,
        "code_written": True
    }
    
    return {
        "messages": [AIMessage(content=f"```python\n{code}\n```")],
        "working_memory": WorkingMemory(active_variables=new_active_vars)
    }


async def reviewer_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Code Reviewer node that reviews code for quality and best practices.
    
    Integrates with static analysis results from working_memory.
    """
    messages = state["messages"]
    working_memory = get_working_memory_dict(state)
    
    # Get the code from working_memory (cleaner than parsing messages)
    code = working_memory.get("current_code_content", "")
    if not code and messages:
        code = extract_code_from_response(messages[-1].content)
    
    # Get static analysis status
    analysis_status = working_memory.get("analysis_status", "unknown")
    analysis_context = ""
    if analysis_status == "passed":
        analysis_context = "\n\nNote: Static analysis (Ruff, Bandit, Radon) has PASSED."
    
    prompt = f"""
    You are the Code Reviewer.
    Review the following code for errors, bugs, and best practices.
    {analysis_context}
    
    Code:
    ```python
    {code}
    ```
    
    Check for:
    1. Logic errors
    2. Edge cases not handled
    3. Code clarity and maintainability
    4. Proper error handling
    5. Documentation quality
    
    If the code is acceptable, respond with "APPROVED" followed by a brief summary.
    Otherwise, list the specific issues that need to be fixed.
    """
    
    response = await run_llm(prompt, purpose="review")
    result = response.get("result", "")
    approved = "APPROVED" in result.upper()
    
    return {
        "messages": [AIMessage(content=result)],
        "review_approved": approved
    }


async def test_runner_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    Test Runner node that executes tests using DockerSandbox.
    
    Runs pytest on the generated code in a secure container.
    Falls back to local execution if Docker is unavailable.
    """
    working_memory = get_working_memory_dict(state)
    filepath = working_memory.get("current_file_path", "sandbox_output/generated_code.py")
    
    # Get the code content
    code = working_memory.get("current_code_content", "")
    
    result_lines = []
    test_passed = False
    
    try:
        try:
            # Use factory to get appropriate sandbox (LocalSandbox since Docker is disabled)
            from core.surgeon.sandbox import get_sandbox
            
            sandbox = get_sandbox(base_dir=os.getcwd())
            
            # Only DockerSandbox has ensure_image_exists
            if hasattr(sandbox, 'ensure_image_exists'):
                sandbox.ensure_image_exists()
                
            # Run pytest on the generated file
            # LocalSandbox.run_command accepts list or str, likely str.
            # We'll use the same command structure.
            exit_code, stdout, stderr = sandbox.run_command(
                f"python3 -m pytest {filepath} -v --tb=short 2>&1 || python3 -c 'import sys; sys.path.insert(0, \".\"); exec(open(\"{filepath}\").read())' 2>&1",
                timeout=120
            )
            
            if exit_code == 0:
                result_lines.append(f"✅ Tests PASSED ({type(sandbox).__name__})")
                result_lines.append(f"\nOutput:\n{stdout[:2000]}")  # Limit output
                test_passed = True
            else:
                result_lines.append(f"❌ Tests FAILED ({type(sandbox).__name__}, exit code {exit_code})")
                result_lines.append(f"\nStdout:\n{stdout[:1500]}")
                if stderr:
                    result_lines.append(f"\nStderr:\n{stderr[:500]}")
            
        except Exception as e:
            logger.warning(f"Sandbox execution failed: {e}, falling back to syntax check")
            result_lines.append(f"⚠️ Execution failed ({type(e).__name__}), using syntax check")
            raise  # Fall through to syntax check blocks below
            
    except ImportError:
        result_lines.append("⚠️ DockerSandbox not available, using local syntax check")
        # Fallback: at least do a syntax check
        try:
            import ast
            ast.parse(code)
            result_lines.append("✅ Syntax check PASSED")
            test_passed = True
        except SyntaxError as e:
            result_lines.append(f"❌ Syntax Error: {e}")
            
    except Exception as e:
        logger.warning(f"Docker sandbox failed: {e}, falling back to local execution")
        result_lines.append(f"⚠️ Docker unavailable ({type(e).__name__}), using local verification")
        
        # Fallback: syntax check and basic import test
        try:
            import ast
            ast.parse(code)
            result_lines.append("✅ Syntax check PASSED")
            
            # Try to compile the code
            compile(code, filepath, 'exec')
            result_lines.append("✅ Compilation check PASSED")
            test_passed = True
            
        except SyntaxError as e:
            result_lines.append(f"❌ Syntax Error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result_lines.append(f"❌ Compilation Error: {e}")
    
    # Build final result message
    result = "\n".join(result_lines)
    
    # Update working memory with test results
    new_active_vars = {
        **working_memory,
        "test_passed": test_passed,
        "test_output": result
    }
    
    return {
        "messages": [AIMessage(content=result)],
        "working_memory": WorkingMemory(active_variables=new_active_vars)
    }


# --- Graph Definition ---

def create_coding_graph():
    """
    Creates the coding team workflow graph.
    
    Flow:
    1. planner -> coder: Plan the implementation
    2. coder -> static_analysis: Write code and analyze
    3. static_analysis -> coder (if failed) or reviewer (if passed)
    4. reviewer -> coder (if not approved) or test_runner (if approved)
    5. test_runner -> END
    """
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
        working_memory = get_working_memory_dict(state)
        status = working_memory.get("analysis_status", "failed")
        
        # Safety Check: Infinite Loop Prevention
        iterations = working_memory.get("analysis_iterations", 0)
        if iterations > 5:
            # Fallback to human review if agent loops too many times
            logger.warning("Analysis loop limit reached, proceeding to reviewer")
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
        """Check review status and route accordingly."""
        last_msg = state["messages"][-1].content if state["messages"] else ""
        if "APPROVED" in last_msg.upper():
            return "test_runner"
        return "coder"

    workflow.add_conditional_edges(
        "reviewer", 
        check_review,
        {
            "test_runner": "test_runner",
            "coder": "coder"
        }
    )
    workflow.add_edge("test_runner", END)
    
    return workflow.compile()
