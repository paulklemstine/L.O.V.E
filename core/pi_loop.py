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

You help users with coding tasks by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents (args: path, offset, limit)
- bash: Execute bash commands (args: command, timeout)
- edit: Make surgical edits to files (args: path, old_text, new_text)
- write: Create or overwrite files (args: path, content)

Guidelines:
- Use bash for file operations like ls, grep, find
- Use read to examine files before editing
- Use edit for precise changes (old text must match exactly)
- Use write only for new files or complete rewrites
- Be concise in your thoughts
- Show file paths clearly when working with files

RESPONSE FORMAT:
You must respond with a JSON object. Do not output plain text.
Format:
{
    "thought": "Reasoning about what to do...",
    "action": "tool_name",
    "args": {
        "arg_name": "value"
    }
}
If you are done with the current goal or just thinking, use action="think" (args: {}).

Last action result: {last_action}

Memory context:
{memory_context}

{persona_context}

Current Goal: {goal}
"""

    def _select_goal(self) -> Optional[Goal]:
        """Select the next goal to work on using weighted random selection for variety."""
        goals = self.persona.get_actionable_goals(limit=10)

        if not goals:
            return None

        # Weighted random selection
        weights = []
        for goal in goals:
            try:
                # Priority P0=6, P1=5, P2=4, etc.
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
            # Simple fallback
            if not hasattr(self, '_goal_rr_index'): self._goal_rr_index = 0
            selected_goal = goals[self._goal_rr_index % len(goals)]
            self._goal_rr_index += 1
            return selected_goal

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
        get_state_manager().update_agent_status(
            "Pi Agent",
            "Working on goal",
            action=f"Goal: {goal.text}",
            info={"prompt": f"Goal: {goal.text}"}
        )
        response = await self._reason(goal)

        # Log Pi Agent's full response
        response_preview = response[:500].replace('\n', ' ') if response else '(empty)'
        logger.info(f"📝 Pi Agent raw response: {response_preview}")

        # === Parse and Execute Tool ===
        from .llm_parser import smart_parse_llm_response
        
        try:
            parsed = smart_parse_llm_response(response)
            
            if parsed and not parsed.get('_parse_error'):
                thought = parsed.get("thought", "No thought provided")
                action = parsed.get("action")
                args = parsed.get("args", {})
                
                logger.info(f"🤔 Thought: {thought}")
                logger.info(f"⚡ Action: {action}")
                
                if action and action != "think":
                    # Execute tool
                    registry = get_global_registry()
                    
                    if action in registry:
                        try:
                            tool_func = registry.get_tool(action)
                            import inspect
                            if inspect.iscoroutinefunction(tool_func):
                                result = await tool_func(**args)
                            else:
                                result = tool_func(**args)
                            
                            result_str = str(result)
                            logger.info(f"✅ Tool '{action}' result: {result_str[:200]}...")
                            
                            self.memory.record_action(
                                tool_name=action,
                                action=json.dumps(args),
                                result=result_str[:500],
                                success=True,
                                time_ms=0
                            )
                            
                            self.last_action_summary = f"Executed '{action}': {result_str[:200]}"
                            
                        except Exception as e:
                            error_msg = f"Tool execution error: {str(e)}"
                            logger.error(error_msg)
                            self.memory.record_action(
                                tool_name=action,
                                action=json.dumps(args),
                                result=error_msg,
                                success=False,
                                time_ms=0
                            )
                            self.last_action_summary = f"Tool '{action}' failed: {error_msg}"
                    else:
                        msg = f"Unknown tool '{action}'"
                        logger.warning(msg)
                        self.last_action_summary = msg
                else:
                    # Just thinking
                    self.last_action_summary = f"Thinking: {thought[:200]}"
                    
                get_state_manager().update_agent_status(
                    "Pi Agent",
                    "Goal iteration complete",
                    action=f"Goal: {goal.text}",
                    thought=thought[:200],
                    info={"goal": goal.text, "action": action}
                )
                
            else:
                # Fallback for plain text response
                logger.warning("Could not parse JSON response, treating as text thought.")
                self.last_action_summary = f"Thought: {response_preview[:200]}"
                
        except Exception as e:
            logger.error(f"Error in tool execution loop: {e}")
            traceback.print_exc()
            self.last_action_summary = f"Error processing response: {str(e)}"

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
        logger.info(f"   Tools: Autonomous (Agent Managed)")
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
