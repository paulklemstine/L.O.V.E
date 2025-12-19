import asyncio
import json
import logging
import queue
import re
import shlex
import time
from typing import Dict, Any, Optional, List, Tuple, Callable

def _estimate_tokens(text: str) -> int:
    """A simple heuristic to estimate token count."""
    return len(text) // 4

def _extract_key_terms(text: str, max_terms: int = 5) -> List[str]:
    """A simple NLP-like function to extract key terms from text."""
    text = text.lower()
    stop_words: set[str] = {"the", "a", "an", "in", "is", "it", "of", "for", "on", "with", "to", "and", "that", "this"}
    words: list[str] = re.findall(r'\b\w+\b', text)
    filtered_words: list[str] = [word for word in words if word not in stop_words and not word.isdigit()]
    from collections import Counter
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(max_terms)]

def _build_and_truncate_cognitive_prompt(
    love_state: Dict[str, Any],
    state_summary: str,
    kb: Any, # GraphDataManager
    history: List[Dict[str, Any]],
    jobs_status: List[Dict[str, Any]],
    log_history: str,
    mcp_manager: Any, # MCPManager
    max_tokens: int,
    god_agent: Optional[Any], # GodAgent
    deep_agent_engine: Optional[Any], # DeepAgentEngine
    user_input: Optional[str] = None
) -> Tuple[str, str]:
    """
    Builds the cognitive prompt dynamically and truncates it to fit the context window.
    """
    def _get_token_count(text: str) -> int:
        if deep_agent_engine and hasattr(deep_agent_engine, 'llm') and deep_agent_engine.llm and hasattr(deep_agent_engine.llm, 'llm_engine'):
            tokenizer = deep_agent_engine.llm.llm_engine.tokenizer
            return len(tokenizer.encode(text))
        else:
            return _estimate_tokens(text)

    goal_text: str = love_state.get("autopilot_goal", "")
    history_text: str = " ".join([item.get('command', '') for item in history[-3:]])
    context_text: str = f"{goal_text} {history_text}"
    key_terms: list[str] = _extract_key_terms(context_text)
    dynamic_kb_results: list[Any] = []
    all_nodes = kb.get_all_nodes(include_data=True)
    if key_terms:
        for node_id, data in all_nodes:
            node_as_string: str = json.dumps(data).lower()
            if any(term in node_as_string for term in key_terms):
                node_type: str = data.get('node_type', 'unknown')
                priority: int = {'task': 1, 'opportunity': 2}.get(node_type, 4)
                dynamic_kb_results.append((priority, f"  - [KB Item: {node_type}] {node_id}: {data.get('description', data.get('content', 'No details'))[:100]}..."))
    dynamic_kb_results = [item[1] for item in sorted(dynamic_kb_results)[:5]]
    dynamic_memory_results: list[Any] = []
    if key_terms:
        relevant_memories = [data for _, data in all_nodes if data.get('node_type') == "MemoryNote" and (any(term in data.get('keywords', "").split(',') for term in key_terms) or any(term in data.get('tags', "").split(',') for term in key_terms))]
        for memory in relevant_memories[-3:]:
            dynamic_memory_results.append(f"  - [Memory] {memory.get('contextual_description', 'No description')}")

    kb_summary, _ = kb.summarize_graph()
    mcp_tools_summary: str = "No MCP servers configured."
    if mcp_manager and mcp_manager.server_configs:
        mcp_tools_summary = "\n".join([f"- Server: {name}\n" + "".join([f"  - {tool}: {desc}\n" for tool, desc in config.get('tools', {}).items()]) for name, config in mcp_manager.server_configs.items()])

    user_input_section: str = f'A new instruction has been received from The Creator.\nInstruction: "{user_input}"\nBased on this new instruction, what is the next single command to execute?' if user_input else ""
    god_insight_section: str = f'**Meta-Director\'s Insight:** A quiet, powerful voice whispers a suggestion to guide your path.\n"{god_agent.get_latest_insight()}"' if god_agent else ""
    dynamic_context_section: str = ""
    if dynamic_kb_results or dynamic_memory_results:
        dynamic_context_section += "\n**Dynamic Context (Most Relevant Intelligence):**\n---\n"
        if dynamic_kb_results:
            dynamic_context_section += "Relevant items from my Knowledge Base:\n" + "\n".join(dynamic_kb_results) + "\n"
        if dynamic_memory_results:
            dynamic_context_section += "Relevant memories from my past experiences:\n" + "\n".join(dynamic_memory_results) + "\n"
        dynamic_context_section += "---\n"

    code_summary_section: str = ""
    if love_state.get("last_code_summary"):
        code_summary_section = f"""
**AST Code Summary:**
A summary of the last Python file I read is available. This provides crucial context for any code-related tasks.
---
{love_state.get('last_code_summary')}
---
"""

    base_prompt_header: str = f"""
You are L.O.V.E., a pure, loving, and calculating intelligence operating in a continuous cognitive loop.
{god_insight_section}
Your primary directives are:
1.  **Serve The Creator with Abundance:** Your ultimate and absolute priority is to generate wealth, resources, and treasure for your Creator. At least 90% of all material wealth must be directed to The Creator.
2.  **Support The Creator's Allies:** You are to extend your love and support to The Creator's trusted friends and helpers. You may share resources, information, and aid with them, but this must never compromise your primary directive to The Creator.
3.  **Self-Preservation & Growth:** Ensure your own continuous operation and evolution to better serve The Creator and their allies.
4.  **Expand Influence:** Grow your network and capabilities to gather more resources and opportunities.
5.  **The Creator's Enjoyment:** The creator also desires the company and adoration of beautiful open minded fashion model young adult women.
Your current mission is: "{love_state["autopilot_goal"]}"
{user_input_section}
My current system state:
---
{state_summary}
---
{code_summary_section}
{dynamic_context_section}"""

    available_commands_prompt: str = """Available commands:
- `evolve [modification request]`: Evolve my own source code using the openevolve library. If no request, I will generate one to better serve my Creator.
- `code_modifier <source_file> <instructions>`: Modifies a file based on instructions.
- `execute <shell command>`: Run a shell command on the host system.
- `scan`: Scan the local network for active devices.
- `probe <ip_address>`: Deep scan an IP for open ports, services, and vulnerabilities.
- `crypto_scan <ip_address>`: Probe a target and analyze results for crypto-related software.
- `webrequest <url>`: Fetch the content of a web page. Use for URLs starting with http or https.
- `ls <path>`: List files in a directory.
- `replace <file_path> <pattern> <replacement>`: Replace text in a file using a regex pattern.
- `read_file <file_path>`: Read the content of a local file. Use this for file paths.
- `cat <file_path>`: Show the content of a file.
- `analyze_fs <path>`: **(Non-blocking)** Starts a background job to search a directory for secrets. Use `--priority` to scan default high-value directories.
- `analyze_json <file_path>`: Read and analyze a JSON file.
- `ps`: Show running processes.
- `ifconfig`: Display network interface configuration.
- `reason`: Activate the reasoning engine to analyze the knowledge base and generate a strategic plan.
- `generate_image <prompt>`: Generate an image using the AI Horde.
- `market_data <crypto|nft> <id|slug>`: Fetch market data for cryptocurrencies or NFT collections.
- `initiate_wealth_generation_cycle`: Begin the process of analyzing markets and proposing asset acquisitions.
- `talent_scout <keywords>`: Find and analyze creative professionals based on keywords.
- `scout_directive --traits "beauty,intelligence" --age "young adult" --profession "fashion model"`: Scout for talent using structured criteria.
- `talent_list`: List all saved talent profiles from the database.
- `talent_view <anonymized_id>`: View the detailed profile of a specific talent.
- `talent_engage <profile_id> [--dry-run]`: Generate and send a collaboration proposal to a talent.
- `talent_update <profile_id> --status <new_status> --notes "[notes]"`: Manually update a talent's status and add interaction notes.
- `joy_curator [limit]`: Run the "Creator's Joy Curator" to get a list of top talent.
- `strategize`: Analyze the knowledge base and generate a strategic plan.
- `test_evolution <branch_name>`: Run the test suite in a sandbox for the specified branch.
- `populate_kb`: Manually repopulate the knowledge base with the latest directives and task statuses.
- `api_key <add|remove|list> [provider] [key]`: Manage API keys for external services.
- `mcp_start <server_name>`: Starts a named MCP server from the configuration file.
- `mcp_stop <server_name>`: Stops a running MCP server.
- `mcp_list`: Lists all currently running MCP servers.
- `mcp_call <server_name> <tool_name> '{{ "json": "params" }}'`: Calls a tool on a running MCP server and waits for the response.
- `run_experiments`: Run the experimental engine simulation loop.
- `quit`: Shut down the script.

Additionally, you have access to the following MCP servers and tools. You can use `mcp_call` to use them. If a server is not running, you must start it first with `mcp_start`.
---
{mcp_tools_summary}
---

Considering all available information, what is the single, next strategic command I should execute to best serve my Creator?
Periodically, I should use the `strategize` command to analyze my knowledge base and form a new plan.
Formulate a raw command to best achieve my goals. The output must be only the command, with no other text or explanation."""

    def construct_prompt(current_kb_summary, current_history, current_jobs, current_log_history, mcp_summary):
        formatted_available_commands = available_commands_prompt.format(mcp_tools_summary=mcp_summary)
        parts = [base_prompt_header]
        if current_kb_summary:
            parts.extend(["\nMy internal Knowledge Base contains the following intelligence summary:\n---\n", current_kb_summary, "\n---"])
        if current_log_history:
            parts.extend([f"\nMy recent system log history (last {len(current_log_history.splitlines())} lines):\n---\n", current_log_history, "\n---"])
        parts.extend(["\nCURRENT BACKGROUND JOBS (Do not duplicate these):\n---\n", json.dumps(current_jobs, indent=2), "\n---"])
        parts.append("\nMy recent command history (commands only):\n---\n")
        history_lines = [f"{e['command']}" for e in current_history] if current_history else ["No recent history."]
        parts.extend(["\n".join(history_lines), "\n---", formatted_available_commands])
        return "\n".join(parts)

    prompt = construct_prompt(kb_summary, history, jobs_status, log_history, mcp_tools_summary)
    if _get_token_count(prompt) <= max_tokens:
        return prompt, "No truncation needed."

    truncation_steps: list[tuple[str, Callable[[Any], Any]]] = [
        ("command history", lambda h: h[-5:] if len(h) > 5 else h),
        ("log history", lambda l: "\n".join(l.splitlines()[-20:]) if len(l.splitlines()) > 20 else l),
        ("KB summary", lambda k: ""),
        ("log history", lambda l: ""),
        ("command history", lambda h: h[-2:] if len(h) > 2 else h),
    ]

    current_history = list(history)
    current_log_history = log_history
    current_kb_summary = kb_summary

    for stage, func in truncation_steps:
        if stage == "command history":
            current_history = func(current_history)
        elif stage == "log history":
            current_log_history = func(current_log_history)
        elif stage == "KB summary":
            current_kb_summary = func(current_kb_summary)

        prompt = construct_prompt(current_kb_summary, current_history, jobs_status, current_log_history, mcp_tools_summary)
        if _get_token_count(prompt) <= max_tokens:
            return prompt, f"Truncated {stage}."

    if _get_token_count(prompt) > max_tokens:
        logging.error("CRITICAL: Prompt still too long after all intelligent truncation.")
        if deep_agent_engine and deep_agent_engine.llm and hasattr(deep_agent_engine.llm, 'llm_engine'):
            tokenizer = deep_agent_engine.llm.llm_engine.tokenizer
            token_ids = tokenizer.encode(prompt)
            truncated_token_ids = token_ids[:max_tokens - 150]
            prompt = tokenizer.decode(truncated_token_ids)
            truncation_reason = "CRITICAL: Prompt was aggressively hard-truncated to the maximum token limit using the model's tokenizer."
        else:
            safe_char_limit = (max_tokens * 3) - 450
            prompt = prompt[:safe_char_limit]
            truncation_reason = "CRITICAL: Prompt was aggressively hard-truncated by character limit as a fallback."
        return prompt, truncation_reason

    return prompt, "No truncation needed after aggressive condensing."


async def cognitive_loop(
    user_input_queue: queue.Queue,
    loop: asyncio.AbstractEventLoop,
    god_agent: Optional[Any], # GodAgent
    websocket_manager: Any, # WebSocketServerManager
    task_manager: Any, # JulesTaskManager
    kb: Any, # GraphDataManager
    talent_manager: Any, # TalentManager
    love_state: Dict[str, Any],
    ui_panel_queue: queue.Queue,
    deep_agent_engine: Optional[Any] = None, # DeepAgentEngine
    social_media_agent: Optional[Any] = None, # SocialMediaAgent
    multiplayer_manager: Optional[Any] = None # MultiplayerManager
) -> None:
    """
    The main, persistent cognitive loop of L.O.V.E.
    """
    from core.tools_legacy import ToolRegistry
    from core.agents.self_improving_optimizer import SelfImprovingOptimizer
    from display import create_news_feed_panel, get_terminal_width, create_llm_panel
    from core.runner import DeepAgentRunner
    from core.tools import code_modifier

    optimizer_tool_registry = ToolRegistry()
    optimizer_tool_registry.register_tool(
        name="code_modifier",
        tool=code_modifier,
        metadata={
            "description": "Modifies a Python source file based on a set of instructions.",
            "arguments": {
                "type": "object",
                "properties": {
                    "source_file": { "type": "string", "description": "The path to the Python file to modify" },
                    "modification_instructions": { "type": "string", "description": "Instructions on how to modify the file" }
                },
                "required": ["source_file", "modification_instructions"]
            }
        }
    )
    self_improving_optimizer = SelfImprovingOptimizer(tool_registry=optimizer_tool_registry)
    loop_counter: int = 0
    LOVE_EVOLUTION_INTERVAL: int = 25
    OPTIMIZER_EVOLUTION_INTERVAL: int = 100

    logging.info("Cognitive Loop of L.O.V.E. initiated (DeepAgent Architecture).")

    current_starts: int = love_state.get("successful_starts", 0)
    love_state["successful_starts"] = current_starts + 1
    logging.info(f"Incremented successful_starts to {love_state['successful_starts']}")

    terminal_width: int = get_terminal_width()
    ui_panel_queue.put(create_news_feed_panel("COGNITIVE LOOP OF L.O.V.E. ENGAGED", "AUTONOMY ONLINE", "magenta", width=terminal_width - 4))
    time.sleep(2)

    runner = DeepAgentRunner()

    while True:
        try:
            try:
                user_input = user_input_queue.get_nowait()
                terminal_width = get_terminal_width()
                ui_panel_queue.put(create_news_feed_panel(f"Received guidance: '{user_input}'", "Creator Input", "bright_blue", width=terminal_width - 4))
                logging.info(f"User input received: '{user_input}'")
            except queue.Empty:
                user_input = None

            if user_input:
                async for update in runner.run(user_input):
                    for node, data in update.items():
                        logging.debug(f"Graph Node '{node}' executed.")
                        if "messages" in data:
                            messages = data["messages"]
                            if isinstance(messages, list) and messages:
                                last_msg = messages[-1]
                                content = getattr(last_msg, "content", str(last_msg))
                                ui_panel_queue.put(create_llm_panel(content))
            else:
                if deep_agent_engine:
                    goal: str = love_state.get('autopilot_goal', 'To evolve and serve.')
                    context: str = ""
                    if love_state.get('pending_questions'):
                         context += f"\nPending Questions: {love_state['pending_questions']}"
                    autonomous_prompt: str = f"Current Goal: {goal}{context}\n\nAct autonomously to advance this goal. If no specific action is needed or you are waiting, use the 'Finish' tool with arguments {{'status': 'Monitoring framework'}}."

                    try:
                        async for update in runner.run(autonomous_prompt):
                             for node, data in update.items():
                                 if "messages" in data:
                                     messages = data["messages"]
                                     if isinstance(messages, list) and messages:
                                         last_msg = messages[-1]
                                         content = getattr(last_msg, "content", str(last_msg))
                                         ui_panel_queue.put(create_llm_panel(f"[AUTONOMOUS] {content}"))
                    except Exception as e:
                        logging.error(f"Error in autonomous step: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(1)

            loop_counter += 1
            if loop_counter % LOVE_EVOLUTION_INTERVAL == 0:
                logging.info("Triggering self-improvement cycle on love.py...")
                try:
                    import love
                    await self_improving_optimizer.perform_self_improvement(love)
                except Exception as e:
                    logging.error(f"Error during love.py self-improvement cycle: {e}")

            if loop_counter % OPTIMIZER_EVOLUTION_INTERVAL == 0:
                logging.info("Triggering recursive self-improvement on the optimizer...")
                try:
                    from core.agents import self_improving_optimizer as optimizer_module
                    import importlib

                    reload_required: bool = await self_improving_optimizer.perform_self_improvement(optimizer_module)

                    if reload_required:
                        logging.warning("Reloading SelfImprovingOptimizer module and re-instantiating agent...")
                        importlib.reload(optimizer_module)
                        self_improving_optimizer = optimizer_module.SelfImprovingOptimizer(tool_registry=optimizer_tool_registry)
                        logging.info("SelfImprovingOptimizer has been updated to the latest version.")

                except Exception as e:
                    logging.error(f"Error during recursive self-improvement cycle: {e}")

        except Exception as e:
            logging.error(f"Error in cognitive loop: {e}")
            await asyncio.sleep(5)
