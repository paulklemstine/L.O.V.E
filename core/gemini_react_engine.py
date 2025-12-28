import inspect
import json
import asyncio
from typing import List, Dict, Any, Tuple

from display import create_reasoning_panel, get_terminal_width
from core.llm_api import execute_reasoning_task
import core.tools_legacy

class GeminiReActEngine:
    """Manages the state of a Thought-Action-Observation loop."""

    def __init__(self, tool_registry: 'core.tools_legacy.ToolRegistry', ui_panel_queue=None, memory_manager=None, caller="Unknown", deep_agent_instance=None):
        self.tool_registry = tool_registry
        self.session_tool_registry = core.tools_legacy.ToolRegistry()
        self.history: List[Tuple[str, str, str]] = []
        self.memory_manager = memory_manager
        self.caller = caller
        self.ui_panel_queue = ui_panel_queue
        self.deep_agent_instance = deep_agent_instance
        self.loop_detection_threshold = 3  # Number of repeated attempts before detecting a loop

    def _log_panel_to_ui(self, panel):
        """Safe method to put a panel into the ui_panel_queue, handling both sync and async queues."""
        if not self.ui_panel_queue:
            return

        item = {"type": "reasoning_panel", "content": panel}
        if isinstance(self.ui_panel_queue, asyncio.Queue):
            # If it's an async queue, we can't await it here easily without making everything async
            # but since this method is called from async execute_goal, we can await it if we change signature.
            # However, to be generic, we use put_nowait which is non-blocking.
            try:
                self.ui_panel_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
        else:
            # Assume it's a synchronous queue.Queue
            self.ui_panel_queue.put(item)

    def _detect_loop(self, tool_name: str, arguments: dict) -> tuple[bool, int, list]:
        """
        Detects if the same tool with similar arguments has been called multiple times recently.
        
        Returns:
            tuple: (is_loop_detected, similar_count, recent_failures)
        """
        if len(self.history) < self.loop_detection_threshold:
            return False, 0, []
        
        # Get recent actions from history
        recent_actions = []
        recent_failures = []
        
        for thought, action, observation in self.history[-(self.loop_detection_threshold * 2):]:
            if isinstance(action, dict) and action.get('tool_name') == tool_name:
                # Check if arguments are similar
                if action.get('arguments') == arguments:
                    recent_actions.append((thought, action, observation))
                    # Collect failures (observations containing "Error" or "failed")
                    if observation and ('error' in observation.lower() or 'failed' in observation.lower()):
                        recent_failures.append(observation)
        
        similar_count = len(recent_actions)
        is_loop = similar_count >= self.loop_detection_threshold
        
        return is_loop, similar_count, recent_failures

    async def execute_goal(self, goal: str, max_steps: int = 10) -> dict:
        """
        Main entry point for the ReAct engine.
        Continues the loop until the Action is a "Finish" action or max_steps is reached.
        Returns a dict with 'success' (bool) and 'result' (str) keys.
        """
        step_count = 0
        while step_count < max_steps:
            step_count += 1
            # Combine static and dynamic tools for the prompt
            main_metadata = self.tool_registry.get_formatted_tool_metadata()
            session_metadata = self.session_tool_registry.get_formatted_tool_metadata()
            tool_metadata = f"{main_metadata}\n{session_metadata}"

            prompt = self._create_prompt(goal, tool_metadata)
            response_dict = await execute_reasoning_task(prompt, deep_agent_instance=self.deep_agent_instance)

            if not response_dict or not response_dict.get("result"):
                # Log the failure
                panel = create_reasoning_panel(
                    caller=self.caller,
                    raw_response="The reasoning engine failed to produce a response.",
                    thought=None, action=None, observation=None,
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(panel)
                return {"success": False, "result": "The reasoning engine failed to produce a response."}

            raw_response = response_dict.get("result", "")

            # Log the raw response
            panel = create_reasoning_panel(
                caller=self.caller,
                raw_response=raw_response,
                thought=None, action=None, observation=None,
                width=get_terminal_width()
            )
            self._log_panel_to_ui(panel)

            # Use smart parser to handle multiple LLM output formats
            from core.llm_parser import smart_parse_llm_response
            
            parsed_response = smart_parse_llm_response(raw_response, expected_keys=["thought", "action"])
            
            # Check if parsing failed
            if parsed_response.get('_parse_error'):
                observation = f"Error: The reasoning engine produced invalid output. {parsed_response['_parse_error']}. Raw response: {parsed_response.get('_raw_response', raw_response[:200])}"
                self.history.append(("Error parsing LLM response", "N/A", observation))
                # Log the parsing error
                panel = create_reasoning_panel(
                    caller=self.caller,
                    raw_response=raw_response,
                    thought="Error parsing LLM response",
                    action=None,
                    observation=observation,
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(panel)
                continue

            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action", {})

            # SAFETY FIX: Ensure action is a dictionary
            if isinstance(action, str):
                try:
                    import ast
                    # Try to parse as JSON or Python dict
                    try:
                        action = json.loads(action)
                    except json.JSONDecodeError:
                        action = ast.literal_eval(action)
                except (ValueError, SyntaxError, Exception):
                    # If parsing fails, log error and skip
                    observation = f"Error: Model produced invalid action format (string that could not be parsed). Action was: {action[:200]}"
                    self.history.append((thought, action, observation))
                    panel = create_reasoning_panel(
                        caller=self.caller,
                        raw_response=None, thought=thought, action=None,
                        observation=observation,
                        width=get_terminal_width()
                    )
                    self._log_panel_to_ui(panel)
                    continue

            if not isinstance(action, dict):
                 # If it's still not a dict (e.g. parsed into a list or something else), fail gracefully
                observation = f"Error: Model produced invalid action format (expected dict, got {type(action)}). Action was: {str(action)[:200]}"
                self.history.append((thought, str(action), observation))
                panel = create_reasoning_panel(
                    caller=self.caller,
                    raw_response=None, thought=thought, action=None,
                    observation=observation,
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(panel)
                continue

            tool_name = action.get("tool_name")
            arguments = action.get("arguments", {})

            # Log the parsed thought and action
            panel = create_reasoning_panel(
                caller=self.caller,
                raw_response=None,
                thought=thought,
                action=action,
                observation=None,
                width=get_terminal_width()
            )
            self._log_panel_to_ui(panel)

            # VALIDATION: Check if tool_name is None or empty
            if not tool_name or tool_name == "None":
                observation = f"Error: Model produced invalid tool_name ('{tool_name}'). This usually indicates the model is confused or stuck. Thought was: {thought[:200]}"
                self.history.append((thought, action, observation))
                # Log the error
                panel = create_reasoning_panel(
                    caller=self.caller,
                    raw_response=None,
                    thought=None,
                    action=None,
                    observation=observation,
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(panel)
                continue

            # LOOP DETECTION: Check if we're repeating the same action
            is_loop, similar_count, recent_failures = self._detect_loop(tool_name, arguments)
            
            if is_loop:
                # Store failure in episodic memory
                if self.memory_manager:
                    failure_context = f"""Reasoning Loop Detected:
Goal: {goal}
Steps Taken: {step_count}
Repeated Action: {tool_name} with arguments {arguments}
Attempts: {similar_count}
Recent Failures:
{chr(10).join([f'- {f[:200]}...' for f in recent_failures[:3]])}
"""
                    try:
                        await self.memory_manager.add_episode(failure_context, tags=['ReasoningFailure', 'Loop', self.caller])
                    except Exception as e:
                        # Don't fail if memory storage fails
                        pass
                
                # Provide clear guidance to try something different
                observation = f"""🔄 LOOP DETECTED: You have tried calling '{tool_name}' with the same arguments {similar_count} times.

Recent failures:
{chr(10).join([f'- {f[:150]}...' for f in recent_failures[:3]])}

⚠️ You MUST try a completely different approach:
1. Use a DIFFERENT tool to accomplish the goal
2. Break down the problem in a NEW way
3. Call 'Finish' if the goal truly cannot be achieved

Do NOT call '{tool_name}' again with the same arguments."""
                
                self.history.append((thought, action, observation))
                
                # Log the loop detection
                panel = create_reasoning_panel(
                    caller=self.caller,
                    raw_response=None,
                    thought=None,
                    action=None,
                    observation=observation,
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(panel)
                continue

            if tool_name == "Finish":
                if arguments:
                    return {"success": True, "result": arguments}
                # No arguments: try to parse JSON from thought string
                try:
                    # Try to parse the thought as JSON
                    import ast
                    # First try json.loads
                    try:
                        parsed_result = json.loads(thought)
                        return {"success": True, "result": parsed_result}
                    except json.JSONDecodeError:
                        # Fallback to ast.literal_eval for Python dict format
                        parsed_result = ast.literal_eval(thought)
                        return {"success": True, "result": parsed_result}
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    # If parsing fails, return the thought string as-is
                    return {"success": True, "result": thought}

            try:
                is_dynamic_tool = False
                # Prioritize session-specific tools over global tools
                if tool_name in self.session_tool_registry.get_tool_names():
                    tool = self.session_tool_registry.get_tool(tool_name)
                    is_dynamic_tool = True
                else:
                    tool = self.tool_registry.get_tool(tool_name)

                # Dependency injection for tools that need the engine instance
                tool_params = inspect.signature(tool).parameters
                if 'engine' in tool_params:
                    arguments['engine'] = self
                if 'parent_engine' in tool_params: # For backwards compatibility
                    arguments['parent_engine'] = self

                # Story 1.3: Emit terminal widget panel for tool visibility
                from display import create_terminal_widget_panel
                import time
                start_time = time.time()
                
                # Emit "executing" status panel
                executing_panel = create_terminal_widget_panel(
                    tool_name=tool_name,
                    arguments=arguments,
                    status="executing",
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(executing_panel)

                if inspect.iscoroutinefunction(tool):
                    observation = await tool(**arguments)
                else:
                    observation = tool(**arguments)

                # Emit "complete" status panel
                elapsed_time = time.time() - start_time
                complete_panel = create_terminal_widget_panel(
                    tool_name=tool_name,
                    status="complete",
                    stdout=str(observation)[:500] if observation else None,
                    elapsed_time=elapsed_time,
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(complete_panel)

                # Story 2.5: Create a ToolMemory note for dynamically discovered tools
                if is_dynamic_tool:
                    import core.shared_state as shared_state
                    tool_memory_content = (
                        f"Dynamically Discovered Tool Usage:\n"
                        f"- Tool Name: {tool_name}\n"
                        f"- Arguments: {json.dumps(arguments, indent=2)}\n"
                        f"- Outcome: {observation}"
                    )
                    # memory_manager.add_episode is async, so we await it.
                    # However, if memory_manager is None or not initialized properly in love.py, this could fail.
                    # In love.py, memory_manager is initialized.
                    # Check if add_episode is async. Yes, it is (read_file confirmed).
                    if shared_state.memory_manager:
                        await shared_state.memory_manager.add_episode(tool_memory_content, tags=['ToolMemory'])

            except Exception as e:
                # Story 1.3: Emit error panel
                from display import create_terminal_widget_panel
                error_panel = create_terminal_widget_panel(
                    tool_name=tool_name,
                    status="error",
                    stderr=str(e),
                    width=get_terminal_width()
                )
                self._log_panel_to_ui(error_panel)
                observation = f"Error executing tool {tool_name}: {e}"

            self.history.append((thought, action, str(observation)))
            
            # Limit history to prevent exponential growth - keep last 10 entries with smart truncation
            if len(self.history) > 10:
                # Keep first 2 and last 8 to preserve early context and recent actions
                self.history = self.history[:2] + self.history[-8:]

            # Log the observation
            panel = create_reasoning_panel(
                caller=self.caller,
                raw_response=None, thought=None, action=None,
                observation=str(observation),
                width=get_terminal_width()
            )
            self._log_panel_to_ui(panel)

        # Max steps reached - store failure in episodic memory
        if self.memory_manager:
            failure_context = f"""Reasoning Max Steps Exceeded:
Goal: {goal}
Steps Taken: {max_steps}
Last {min(5, len(self.history))} actions:
{chr(10).join([f'- Thought: {t[:100]}... Action: {str(a)[:50]}... -> Observation: {o[:100]}...' for t, a, o in self.history[-5:]])}
Reason: Maximum reasoning steps exceeded without reaching goal
"""
            try:
                await self.memory_manager.add_episode(failure_context, tags=['ReasoningFailure', 'MaxSteps', self.caller])
            except Exception:
                pass
        
        return {"success": False, "result": f"Goal failed. The reasoning engine exceeded the maximum number of steps ({max_steps})."}

    def _create_prompt(self, goal: str, tool_metadata: str) -> str:
        """Creates the ReAct prompt template."""

        # Story 3.3: Integrate Folded Memories
        relevant_wisdom = ""
        if self.memory_manager:
            # retrieve_relevant_folded_memories is synchronous
            folded_memories = self.memory_manager.retrieve_relevant_folded_memories(goal)
            if folded_memories:
                relevant_wisdom = "You have the following high-level summaries from past experiences that may be relevant:\n"
                for mem in folded_memories:
                    relevant_wisdom += f"- {mem}\n"
                relevant_wisdom += "\n"

        from core.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        
        prompt = registry.render_prompt(
            "react_reasoning",
            goal=goal,
            tool_metadata=tool_metadata,
            relevant_wisdom=relevant_wisdom,
            history=self.history
        )
        
        # Apply prompt compression if applicable
        from core.prompt_compressor import compress_prompt, should_compress
        
        if should_compress(prompt, purpose="react_reasoning"):
            # Force tokens to preserve critical ReAct structure
            force_tokens = [
                "thought", "action", "tool_name", "arguments", "Finish",
                "decompose_and_solve_subgoal", "talent_scout", "opportunity_scout"
            ]
            
            compression_result = compress_prompt(
                prompt,
                force_tokens=force_tokens,
                purpose="react_reasoning"
            )
            
            if compression_result["success"]:
                prompt = compression_result["compressed_text"]
                # Note: We don't log here to avoid cluttering the output during ReAct loops
        
        return prompt
