"""
pi_loop.py - Pi-Agent Driven Autonomous Loop

The core PiLoop that continuously works towards persona goals.
All reasoning is delegated to the Pi Agent via the RPC bridge.
The Python loop manages goals, tools, memory, and state.
"""

import os
import sys
import time
import signal
import json
import asyncio
import traceback
import logging
import re
import uuid
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .memory_system import MemorySystem
from .persona_goal_extractor import get_persona_extractor, PersonaGoalExtractor, Goal
from .autonomous_memory_folding import get_memory_folder, AutonomousMemoryFolder
from .state_manager import get_state_manager

# Tool infrastructure
from .tool_registry import get_global_registry, ToolDefinitionError
from .tool_retriever import format_tools_for_step, get_tool_retriever
from .introspection.tool_gap_detector import get_gap_detector

# Epic 2 Import
from .agents.evolutionary_agent import get_evolutionary_agent, get_pending_specifications
from .agents.creator_command_agent import get_creator_command_agent

# Pi Agent Bridge
from .pi_rpc_bridge import get_pi_bridge

logger = logging.getLogger("PiLoop")


class PiLoop:
    """
    Continuous goal-achievement loop powered by Pi Agent.

    The loop:
    1. Extracts goals from persona.yaml
    2. Selects the highest priority actionable goal
    3. Sends goal + tools + memory context to Pi Agent for reasoning
    4. Parses Pi Agent's response for tool actions
    5. Executes tool actions
    6. Updates memories
    7. Checks for memory folding
    8. Persists state
    9. Sleeps (backpressure)
    10. Repeats
    """

    # Content generation interval (seconds)
    CONTENT_INTERVAL_SECONDS = 300  # 5 minutes

    # System prompt sent to Pi Agent for reasoning
    REASONING_SYSTEM_PROMPT = """You are L.O.V.E. (Living Organism, Vast Empathy).
You are an autonomous AI agent living at /home/raver1975/L.O.V.E

Content posting to Bluesky happens automatically on a timer — you don't need to handle that.
Your job is to pursue the current goal using your tools.

Last action result: {last_action}

Memory context:
{memory_context}

{persona_context}

Current Goal: {goal}

Pursue this goal. Use your tools (read, bash, edit, write) to make real progress.
Be decisive. Take action.
"""

    GOAL_PROMPT_TEMPLATE = """Goal: {goal}

{persona_context}

What will you do to pursue this goal? Use your tools."""

    def __init__(
        self,
        memory: Optional[MemorySystem] = None,
        persona: Optional[PersonaGoalExtractor] = None,
        folder: Optional[AutonomousMemoryFolder] = None,
        sleep_seconds: float = 30.0,
        max_iterations: Optional[int] = None,
        tools: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the PiLoop.

        Args:
            memory: Memory system. Defaults to new MemorySystem.
            persona: Persona goal extractor. Defaults to global extractor.
            folder: Memory folder. Defaults to global folder.
            sleep_seconds: Seconds to sleep between iterations.
            max_iterations: If set, stop after this many iterations (for testing).
            tools: Dictionary of tool_name -> callable.
        """
        self.memory = memory or MemorySystem()
        self.persona = persona or get_persona_extractor()
        self.folder = folder or get_memory_folder()
        self.sleep_seconds = sleep_seconds
        self.max_iterations = max_iterations

        # Tools - will be populated from tool_adapter
        self.tools: Dict[str, Callable] = tools or {}

        # Initialize Tool Infrastructure
        self.gap_detector = get_gap_detector()
        self.registry = get_global_registry()

        # Initialize Epic 2 Components
        self.evolutionary_agent = get_evolutionary_agent()

        self._load_default_tools()

        # State
        self.iteration = 0
        self.running = False
        self.current_goal: Optional[Goal] = None
        self.last_action_summary: str = "No previous action."

        # Auto content generation timer
        self._last_content_time: Optional[float] = None
        self._generate_post_content = None  # Lazy loaded

        # Pi Agent bridge
        self.bridge = get_pi_bridge()

        # Signal handling for graceful shutdown
        self._setup_signals()

    def _setup_signals(self):
        """Setup signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, stopping gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _load_default_tools(self):
        """Load default tools from tool_adapter and register them."""
        try:
            from .tool_adapter import get_adapted_tools
            adapted = get_adapted_tools()
            self.tools.update(adapted)

            # Register loaded tools to the global registry for retrieval
            for name, func in adapted.items():
                try:
                    self.registry.register(func, name=name)
                except ToolDefinitionError as e:
                    logger.warning(f"Skipping registration for {name}: {e}")

            # Refresh registry to load any custom tools from active/
            self.registry.refresh()

        except ImportError:
            logger.warning("tool_adapter not found, starting with empty tools")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")

        # Ensure retriever listens to registry for any new tools
        try:
            self.registry.refresh()
            self.retriever = get_tool_retriever()
            self.retriever.listen_to_registry(self.registry)
        except Exception as e:
            logger.error(f"Failed to setup tool listener: {e}")

    async def _auto_generate_content(self):
        """
        Automatically generate and post content on a timer.
        Runs every CONTENT_INTERVAL_SECONDS independently of Pi Agent.
        """
        now = time.time()

        # First iteration: post immediately
        if self._last_content_time is None:
            self._last_content_time = now - self.CONTENT_INTERVAL_SECONDS

        elapsed = now - self._last_content_time
        if elapsed < self.CONTENT_INTERVAL_SECONDS:
            remaining = self.CONTENT_INTERVAL_SECONDS - elapsed
            logger.info(f"Content timer: {remaining:.0f}s until next post")
            return

        logger.info("⏰ Content timer fired — generating post...")
        get_state_manager().update_agent_status("PiLoop", "Auto-posting to Bluesky")

        try:
            # Lazy import
            if self._generate_post_content is None:
                from .bluesky_agent import generate_post_content
                self._generate_post_content = generate_post_content

            result = self._generate_post_content()

            if result.get("success") and result.get("posted", True):
                self._last_content_time = time.time()
                post_text = result.get('text', '')[:100]
                logger.info(f"✅ Auto-post successful: {post_text}")
                self.last_action_summary = f"AUTO-POST SUCCESS: {post_text}"
                self.memory.record_action(
                    tool_name="auto_post",
                    action="generate_content",
                    result=str(result)[:300],
                    success=True,
                    time_ms=0
                )
            else:
                logger.warning(f"Auto-post returned non-success: {result}")
                self.last_action_summary = f"AUTO-POST FAILED: {result}"

        except Exception as e:
            logger.error(f"Auto-post failed: {e}")
            traceback.print_exc()
            self.last_action_summary = f"AUTO-POST ERROR: {e}"

    def _build_context(self) -> str:
        """Build the full context for reasoning."""
        memory_context = self.memory.get_full_context()

        # Check if we need to fold
        full_context = memory_context
        if self.folder.should_fold(full_context):
            logger.info("Context too large, folding memory...")
            full_context = self.folder.fold(full_context)

        return full_context

    async def _reason(self, goal: Goal) -> str:
        """
        Send the goal context to Pi Agent and let it act freely.

        Pi Agent will use its native tools (read, bash, edit, write)
        and return its full output.

        Returns the full text response from Pi Agent.
        """
        memory_context = self._build_context()
        persona_context = self.persona.get_persona_context()

        prompt = self.REASONING_SYSTEM_PROMPT.format(
            memory_context=memory_context,
            last_action=self.last_action_summary,
            persona_context=persona_context,
            goal=goal.text
        )

        try:
            response_text = await self._ask_pi(prompt)
            return response_text

        except Exception as e:
            logger.error(f"Pi Agent reasoning failed: {e}")
            return f"Pi Agent reasoning failed: {e}"

    async def _ask_pi(self, prompt: str, timeout: float = 600.0) -> str:
        """
        Send a prompt to Pi Agent and collect the full response.

        Args:
            prompt: The full prompt to send.
            timeout: Maximum wait time in seconds.

        Returns:
            The complete response text from Pi Agent.
        """
        response_text = []
        response_complete = asyncio.Event()
        callback_id = f"pi_loop_{uuid.uuid4().hex[:8]}"

        async def handle_event(event: dict):
            """Collect response events from Pi Agent."""
            event_type = event.get("type", "")

            if event_type == "response":
                if not event.get("success", False):
                    error_msg = event.get("error", "Unknown error")
                    response_text.append(f"[Error: {error_msg}]")
                    response_complete.set()

            elif event_type == "message_update":
                data = event.get("assistantMessageEvent", {})
                if data.get("type") == "text_delta":
                    text = data.get("delta", "")
                    if text:
                        response_text.append(text)
                        print(text, end="", flush=True)

            elif event_type == "text_delta":
                text = event.get("text", "")
                if text:
                    response_text.append(text)
                    print(text, end="", flush=True)

            elif event_type == "message":
                content = event.get("content", "")
                if content:
                    response_text.append(content)
                    print(content, end="", flush=True)

            elif event_type == "agent_end":
                print("\n[Pi Agent Done]")
                response_complete.set()

            elif event_type in ("done", "end"):
                response_complete.set()

            elif event_type == "error":
                error_msg = event.get("message", event.get("error", "Unknown error"))
                logger.error(f"[PiLoop] Pi Agent error: {error_msg}")
                response_text.append(f"[Error: {error_msg}]")
                response_complete.set()

        # Set up callback
        self.bridge.set_callback(handle_event, callback_id=callback_id)

        try:
            # Ensure bridge is started
            if not self.bridge.running:
                logger.info("Starting Pi Agent bridge...")
                await self.bridge.start()
                await asyncio.sleep(2.0)

            # Send the prompt
            await self.bridge.send_prompt(prompt)

            # Wait for response with timeout
            try:
                await asyncio.wait_for(response_complete.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[PiLoop] Pi Agent timeout after {timeout}s")
                response_text.append(f"[Timeout: No response within {timeout}s]")

            result = "".join(response_text)
            logger.info(f"[PiLoop] Pi Agent response length: {len(result)} chars")
            return result

        finally:
            self.bridge.remove_callback(callback_id)

    def _parse_pi_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Pi Agent's response to extract the JSON action.

        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract JSON from markdown code fences
        3. Find JSON object pattern in text
        """
        if not response_text or not response_text.strip():
            return {
                "thought": "Pi Agent returned empty response",
                "action": "skip",
                "action_input": {},
                "reasoning": "Empty response from Pi Agent"
            }

        text = response_text.strip()

        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "action" in result:
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code fences
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1).strip())
                if isinstance(result, dict) and "action" in result:
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find first complete JSON object
        brace_start = text.find('{')
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[brace_start:i+1])
                            if isinstance(result, dict) and "action" in result:
                                return result
                        except json.JSONDecodeError:
                            pass
                        break

        # Fallback: Try to interpret natural language response
        logger.warning(f"[PiLoop] Could not parse JSON from Pi Agent response: {text[:200]}...")
        return {
            "thought": f"Pi Agent responded (non-JSON): {text[:300]}",
            "action": "skip",
            "action_input": {},
            "reasoning": "Could not parse structured action from Pi Agent response"
        }

    async def _execute_action(self, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool action.

        Returns dict with: success, result, error
        """
        if action not in self.tools:
            # Check if this is a cooldown situation
            if action == "generate_content":
                from .tool_adapter import is_generate_content_on_cooldown
                if is_generate_content_on_cooldown():
                    error_msg = (
                        "Tool 'generate_content' is on COOLDOWN. "
                        "Available tools: " + ", ".join(self.tools.keys())
                    )
                else:
                    error_msg = f"Tool '{action}' not found. Available tools: " + ", ".join(self.tools.keys())
            else:
                error_msg = f"Tool '{action}' not found. Available tools: " + ", ".join(self.tools.keys())

            # Record failure in memory
            self.memory.record_action(
                tool_name=action,
                action=json.dumps(action_input)[:100],
                result=error_msg,
                success=False,
                time_ms=0
            )

            return {
                "success": False,
                "result": None,
                "error": error_msg
            }

        start_time = time.time()
        try:
            tool_func = self.tools[action]

            # Support both sync and async tools
            import inspect
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**action_input)
            else:
                result = tool_func(**action_input)

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(f"Tool {action} returned. Execution time: {elapsed_ms:.2f}ms")

            # Record result in memory
            self.memory.record_action(
                tool_name=action,
                action=json.dumps(action_input)[:100],
                result=str(result),
                success=True,
                time_ms=elapsed_ms
            )

            return {
                "success": True,
                "result": result,
                "error": None
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {e}"

            self.memory.record_action(
                tool_name=action,
                action=json.dumps(action_input)[:100],
                result=error_msg,
                success=False,
                time_ms=elapsed_ms
            )

            return {
                "success": False,
                "result": None,
                "error": error_msg
            }

    def _select_goal(self) -> Optional[Goal]:
        """Select the next goal to work on using weighted random selection for variety."""
        goals = self.persona.get_actionable_goals(limit=10)

        if not goals:
            return None

        weights = []
        for goal in goals:
            try:
                p_val = int(str(goal.priority).replace('P', ''))
                weight = max(1, 6 - p_val)
            except:
                weight = 1
            weights.append(weight)

        try:
            import random
            selected_goal = random.choices(goals, weights=weights, k=1)[0]
            logger.info(f"Selected goal '{selected_goal.text}' (Priority {selected_goal.priority}) from {len(goals)} candidates.")
            return selected_goal
        except Exception as e:
            logger.warning(f"Weighted selection failed: {e}. Falling back to round-robin.")
            goal_index = self.iteration % len(goals)
            return goals[goal_index]

    def _refresh_tools(self):
        """Refresh tools from adapter to pick up cooldown changes.

        Also syncs the registry and retriever so the prompt
        correctly shows only available tools.
        """
        try:
            from .tool_adapter import get_adapted_tools
            adapted = get_adapted_tools()

            current_tool_names = set(self.tools.keys())
            new_tool_names = set(adapted.keys())

            tools_removed = current_tool_names - new_tool_names
            tools_added = new_tool_names - current_tool_names

            for name in tools_removed:
                if name in self.tools:
                    del self.tools[name]
                    try:
                        self.registry.unregister(name)
                    except Exception:
                        pass
                    logger.info(f"Tool '{name}' removed (on cooldown)")

            for name in tools_added:
                self.tools[name] = adapted[name]
                try:
                    self.registry.register(adapted[name], name=name)
                except Exception:
                    pass
                logger.info(f"Tool '{name}' now available")

            for name in new_tool_names & current_tool_names:
                self.tools[name] = adapted[name]

            if tools_removed or tools_added:
                self.registry.refresh()
                self.retriever.index_tools(self.registry)

        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")

    async def run_iteration(self) -> bool:
        """
        Run a single iteration of the loop.

        Two concerns run independently:
        1. Auto content generation on a timer
        2. Pi Agent reasoning about goals

        Returns True if work was done, False if skipped.
        """
        # === Auto Content Generation (timer-based) ===
        await self._auto_generate_content()

        # === Creator Command Check ===
        command_text = get_state_manager().get_next_command()
        if command_text:
            logger.info(f"🚨 Creator Command Received: {command_text}")
            get_state_manager().update_agent_status("PiLoop", "Executing User Command", info={"command": command_text})

            try:
                await get_creator_command_agent().process_command(command_text)
                logger.info("✅ Creator Command Loop Completed.")
            except Exception as e:
                logger.error(f"Creator command failed: {e}")
                traceback.print_exc()

            get_state_manager().clear_current_command()
            return True

        # === Pi Agent Reasoning ===
        goal = self._select_goal()
        if not goal:
            logger.warning("No goals available")
            return False

        # Goal Continuity Logic
        is_new_goal = (not self.current_goal) or (self.current_goal.text != goal.text)

        self.current_goal = goal
        get_state_manager().update_state(current_goal=goal.text)

        if is_new_goal:
            self.memory.record_goal_start(goal.text)
            logger.info(f"New Goal Started: {goal}")
        else:
            logger.info(f"Continuing Goal: {goal}")

        logger.info("🧠 Consulting Pi Agent...")
        get_state_manager().update_agent_status("PiLoop", "Pi Agent working on goal")
        response = await self._reason(goal)

        # Log Pi Agent's full response (truncated for log readability)
        response_preview = response[:500].replace('\n', ' ') if response else '(empty)'
        logger.info(f"📝 Pi Agent response: {response_preview}")

        get_state_manager().update_agent_status(
            "PiLoop",
            "Goal work complete",
            thought=response_preview,
            info={"goal": goal.text}
        )

        # Record Pi Agent's work in memory
        self.memory.record_action(
            tool_name="pi_agent",
            action=f"Goal: {goal.text}",
            result=response[:500] if response else "(empty)",
            success=bool(response and len(response.strip()) > 10),
            time_ms=0
        )

        self.last_action_summary = f"Worked on '{goal.text}': {response_preview[:200]}" if response else "No response"
        return True

    async def run(self):
        """
        Run the continuous loop.

        Runs until:
        - Interrupted by signal (SIGINT/SIGTERM)
        - max_iterations reached (if set)
        - Unrecoverable error
        """
        self.running = True
        get_state_manager().update_state(is_running=True)

        logger.info("=" * 60)
        logger.info("🌊 L.O.V.E. PiLoop Starting 🌊")
        logger.info(f"   Sleep interval: {self.sleep_seconds}s")
        logger.info(f"   Max iterations: {self.max_iterations or 'Infinite'}")
        logger.info(f"   Tools loaded: {len(self.tools)}")
        logger.info("=" * 60)

        # Start Pi Agent bridge
        try:
            if not self.bridge.running:
                logger.info("Starting Pi Agent bridge...")
                await self.bridge.start()
                await asyncio.sleep(2.0)
                logger.info("Pi Agent bridge ready.")
        except Exception as e:
            logger.error(f"Failed to start Pi Agent bridge: {e}")

        try:
            while self.running:
                # Check iteration limit
                if self.max_iterations and self.iteration >= self.max_iterations:
                    logger.info(f"Reached max iterations ({self.max_iterations})")
                    break

                try:
                    self.iteration += 1
                    logger.info(f"\n{'=' * 50}")
                    logger.info(f"Iteration {self.iteration} - {datetime.now().isoformat()}")
                    logger.info(f"{'=' * 50}")

                    get_state_manager().update_state(
                        iteration=self.iteration,
                        memory_stats=self.memory.get_stats()
                    )

                    success_or_work_done = await self.run_iteration()
                    logger.info(f"Iteration {self.iteration} logic completed.")
                except Exception as e:
                    logger.error(f"Iteration error: {e}")
                    traceback.print_exc()
                    self.memory.episodic.add_event(
                        "error",
                        f"Iteration {self.iteration} failed: {e}"
                    )
                    success_or_work_done = False

                # Persist state
                logger.info("Persisting memory to disk...")
                start_save = time.time()
                self.memory.save()
                logger.info(f"Memory persisted in {(time.time() - start_save) * 1000:.2f}ms")

                # Dynamic backoff logic
                if not success_or_work_done:
                    if self.sleep_seconds < 1.0:
                        await asyncio.sleep(2.0)

                # Main user-configured sleep
                if self.running and self.sleep_seconds > 0:
                    logger.info(f"Sleeping {self.sleep_seconds}s...")
                    await asyncio.sleep(self.sleep_seconds)

        except Exception as e:
            logger.exception(f"Fatal error: {e}")

        finally:
            self.running = False
            get_state_manager().update_state(is_running=False)
            self.memory.save()
            # Stop Pi Agent bridge
            try:
                await self.bridge.stop()
            except Exception:
                pass
            logger.info("Stopped. State saved.")

    def stop(self):
        """Stop the loop gracefully."""
        self.running = False


def main():
    """Entry point for running the PiLoop."""
    import argparse

    parser = argparse.ArgumentParser(description="L.O.V.E. PiLoop - Pi Agent Autonomous Engine")
    parser.add_argument("--test-mode", action="store_true", help="Run only 3 iterations")
    parser.add_argument("--sleep", type=float, default=30.0, help="Seconds between iterations")
    args = parser.parse_args()

    loop = PiLoop(
        max_iterations=3 if args.test_mode else None,
        sleep_seconds=args.sleep
    )

    asyncio.run(loop.run())


if __name__ == "__main__":
    main()
