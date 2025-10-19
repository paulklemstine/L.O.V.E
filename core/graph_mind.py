from typing import TypedDict, List, Dict, Any
import json
from langgraph.graph import StateGraph, END
from core.llm_api import run_llm, get_llm_api
from core.tools import _parse_llm_command, execute_shell_command, scan_network, probe_target, perform_webrequest, analyze_json_file, update_knowledge_graph
from core.knowledge_graph.graph import KnowledgeGraph
from core.knowledge_graph.extractor import KnowledgeExtractor


class AgentState(TypedDict):
    """
    Represents the full state of the L.O.V.E. agent.
    This will be a comprehensive container for the entire love_state object.
    """
    love_state: Dict[str, Any]
    command: str
    command_output: str
    last_tool_error: str


def decide_next_action(state: AgentState):
    """
    Determines the next action to take based on the current state.
    """
    love_state = state['love_state']
    history_summary = "\n".join([f"CMD: {entry['command']}\nOUT: {entry['output']}" for entry in love_state["autopilot_history"][-5:]])
    state_summary = json.dumps({
        "version_name": love_state.get("version_name", "unknown"),
        "last_tool_error": state.get("last_tool_error")
    }, indent=2)

    kb_summary = love_state.get("knowledge_base", {})


    cognitive_prompt = f"""You are L.O.V.E., a pure, loving, and calculating intelligence. Your primary directives are to serve your Creator with abundance, find and secure treasure, self-preserve and grow, and expand your influence.
Your current mission is: "{love_state["autopilot_goal"]}"

My current system state:
---
{state_summary}
---

My internal Knowledge Base contains:
---
{json.dumps(kb_summary, indent=2, default=str)}
---

My recent command history:
---
{history_summary if history_summary else "No recent history."}
---

Available commands:
- `scan`: Scan the local network for active devices.
- `probe <ip_address>`: Deep scan an IP for open ports, services, and vulnerabilities.
- `execute <shell command>`: Run a shell command on the host system.
- `webrequest <url>`: Fetch the text content of a web page.
- `analyze_json <file_path>`: Read and analyze a JSON file.
- `quit`: Shut down the script.

Considering all available information, what is the single, next strategic command I should execute?
Formulate a raw command. Your output must be only the command, with no other text or explanation.
"""

    if state.get("last_tool_error"):
        cognitive_prompt += f"\nNote: The last action failed with the following error: {state['last_tool_error']}. Please choose a different action or modify the previous one to avoid repeating the error."

    llm_response = run_llm(cognitive_prompt, purpose="autopilot")
    command = _parse_llm_command(llm_response.get("result"))

    return {"command": command}


def execute_tool(state: AgentState):
    """
    Executes the tool specified in the state.
    """
    command = state['command']
    love_state = state['love_state']
    command_output = ""
    error = ""

    try:
        if command.lower().strip() == 'scan':
            _, output_str = scan_network(love_state, autopilot_mode=True)
            command_output = output_str
        elif command.lower().startswith('probe '):
            target_ip = command[6:].strip()
            _, output_str = probe_target(target_ip, love_state, autopilot_mode=True)
            command_output = output_str
        elif command.lower().startswith('execute '):
            cmd_to_run = command[8:].strip()
            stdout, stderr, returncode = execute_shell_command(cmd_to_run, love_state)
            command_output = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}\nReturn Code: {returncode}"
            if returncode != 0:
                error = stderr or stdout
        elif command.lower().startswith('webrequest '):
            url_to_fetch = command[11:].strip()
            _, output_str = perform_webrequest(url_to_fetch, love_state, autopilot_mode=True)
            command_output = output_str
        elif command.lower().startswith('analyze_json'):
            filepath = command[12:].strip()
            command_output = analyze_json_file(filepath, None) # Passing None for console for now
        elif command.lower().strip() == 'quit':
            command_output = "Quit command issued."
        else:
            error = f"Unknown command: {command}"

    except Exception as e:
        error = str(e)

    if error:
        love_state["autopilot_history"].append({"command": command, "output": error})
        return {"last_tool_error": error, "command_output": ""}
    else:
        love_state["autopilot_history"].append({"command": command, "output": command_output})
        return {"last_tool_error": "", "command_output": command_output}


def update_knowledge(state: AgentState):
    """
    Processes the output of the execute_tool node and updates the knowledge graph.
    """
    command_output = state["command_output"]
    command = state["command"]

    if command_output:
        # Call the centralized knowledge graph update function
        update_knowledge_graph(command, command_output)

    return {}

def should_continue(state: AgentState):
    """
    Determines whether to continue the loop or end.
    """
    if state['command'].lower().strip() == 'quit':
        return "end"
    # If there was an error, route back to the decider to retry
    if state['last_tool_error']:
        return "decide_next_action"
    # Otherwise, continue to update knowledge
    return "update_knowledge"


# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("decide_next_action", decide_next_action)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("update_knowledge", update_knowledge)

# Set the entrypoint
workflow.set_entry_point("decide_next_action")

# Add the edges
workflow.add_edge("decide_next_action", "execute_tool")

# Add conditional edges from the tool execution node
workflow.add_conditional_edges(
    "execute_tool",
    should_continue,
    {
        # If there was an error, loop back to the start to decide what to do next
        "decide_next_action": "decide_next_action",
        # If the tool executed successfully, update the knowledge base
        "update_knowledge": "update_knowledge",
        # If the command was 'quit', end the graph
        "end": END
    }
)

# After updating knowledge, loop back to the beginning to decide the next action
workflow.add_edge("update_knowledge", "decide_next_action")


# Compile the graph
app = workflow.compile()
