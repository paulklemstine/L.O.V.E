# This module will house the logic for creating and managing Microsoft Agent Framework agents and workflows.
import inspect
import json
from typing import Callable, List, Dict, Any
from agent_framework import ai_function, WorkflowBuilder, executor, WorkflowContext
from core.tools import ToolRegistry
from core.llm_api import run_llm
import asyncio

def _create_tool_wrapper(tool_name: str, tool_callable: Callable, metadata: Dict[str, Any]) -> Callable:
    """
    Dynamically creates a wrapper function for a given tool that is compatible
    with the Microsoft Agent Framework's schema generation.
    """
    description = metadata.get("description", "No description available.")

    sig_params = []
    for param_name, schema in metadata.get("arguments", {}).get("properties", {}).items():
        param = inspect.Parameter(
            param_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str
        )
        sig_params.append(param)

    signature = inspect.Signature(parameters=sig_params)

    async def wrapper(*args, **kwargs):
        if inspect.iscoroutinefunction(tool_callable):
            return await tool_callable(*args, **kwargs)
        else:
            return tool_callable(*args, **kwargs)

    wrapper.__name__ = tool_name
    wrapper.__doc__ = description
    wrapper.__signature__ = signature

    return ai_function(wrapper)

def get_maf_tools(tool_registry: ToolRegistry) -> List[Callable]:
    """
    Creates MAF-compatible wrappers for all tools in the provided ToolRegistry.
    """
    maf_tools = []
    for name, data in tool_registry.list_tools().items():
        wrapped_tool = _create_tool_wrapper(name, data["tool"], data["metadata"])
        maf_tools.append(wrapped_tool)

    return maf_tools

async def create_and_run_workflow(task: str, tool_registry: ToolRegistry) -> str:
    """
    Dynamically creates and runs a multi-agent workflow to accomplish a complex task.
    """
    maf_tools = get_maf_tools(tool_registry)

    # Step 1: Use LLM to generate the workflow plan
    planning_prompt = f"""
You are a master workflow architect. Based on the following complex task, design a multi-agent workflow.
Your output must be a JSON object with two keys: "agents" and "workflow".

The "agents" key should be a list of agent definitions, each with:
- "name": A unique name for the agent (e.g., "researcher").
- "instructions": The system message for the agent.

The "workflow" key should be a list of steps, each with:
- "from": The name of the agent sending the message.
- "to": The name of the agent receiving the message.
The first step should always have "user" as the "from" field. The final step should have "user" as the "to" field.

Task: "{task}"

Example:
Task: "Research the current price of Bitcoin and write a short summary."
{{
  "agents": [
    {{
      "name": "researcher",
      "instructions": "You are a research agent. You find information on the web. Use the 'execute' tool with curl to get data from a crypto API."
    }},
    {{
      "name": "writer",
      "instructions": "You are a writing agent. You summarize information."
    }}
  ],
  "workflow": [
    {{ "from": "user", "to": "researcher" }},
    {{ "from": "researcher", "to": "writer" }},
    {{ "from": "writer", "to": "user" }}
  ]
}}

Now, design the workflow for the given task.
"""

    plan_dict = await run_llm(planning_prompt, purpose="workflow_planning", force_model=None)
    try:
        plan = json.loads(plan_dict.get("result", "{}"))
    except json.JSONDecodeError:
        return "Error: The LLM failed to generate a valid JSON workflow plan."

    # Step 2: Create Executor instances from the plan
    executors = {}
    for agent_def in plan.get("agents", []):
        agent_name = agent_def["name"]

        @executor(id=agent_name)
        async def agent_executor(input_msg: str, ctx: WorkflowContext[str, str]) -> None:
            current_agent_name = ctx.executor_id
            current_agent_def = next((a for a in plan.get("agents", []) if a["name"] == current_agent_name), None)
            if not current_agent_def:
                await ctx.yield_output(f"Error: Agent definition for '{current_agent_name}' not found.")
                return

            agent_prompt = f"""
{current_agent_def['instructions']}
You have received the following message:
---
{input_msg}
---
You have access to the following tools:
{tool_registry.get_formatted_tool_metadata()}
Based on the message and your instructions, determine the next step.
You can either call a tool or respond with a message to the next agent.
Your response should be the tool call in the specified JSON format or a natural language message.
"""
            response_dict = await run_llm(agent_prompt, purpose="agent_execution", force_model=None)
            response_text = response_dict.get("result", "I am unable to proceed.")

            is_final_step = any(step for step in plan.get("workflow", []) if step["from"] == current_agent_name and step["to"] == "user")
            if is_final_step:
                await ctx.yield_output(response_text)
            else:
                await ctx.send_message(response_text)

        executors[agent_name] = agent_executor

    # Step 3: Build the workflow graph
    builder = WorkflowBuilder()
    workflow_steps = plan.get("workflow", [])
    if not workflow_steps:
        return "Error: The LLM plan did not contain any workflow steps."

    start_node = None
    for step in workflow_steps:
        from_node = step["from"]
        to_node = step["to"]

        if from_node == "user":
            start_node = executors.get(to_node)
            continue

        if to_node == "user":
            continue

        from_exec = executors.get(from_node)
        to_exec = executors.get(to_node)

        if from_exec and to_exec:
            builder.add_edge(from_exec, to_exec)

    if not start_node:
        return "Error: Could not determine the start node of the workflow."

    builder.set_start_executor(start_node)
    workflow = builder.build()

    # Step 4: Run the workflow
    final_result = await workflow.run(task)
    return final_result
