"""
deep_loop.py - Main Autonomous Reasoning Loop

The core DeepLoop that continuously works towards persona goals.
Implements unified agentic reasoning with tool discovery and execution.

See docs/deep_loop.md for detailed documentation.
"""

import os
import sys
import time
import signal
import json
import traceback
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path

# Add parent to path for L.O.V.E. v1 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .llm_client import get_llm_client, LLMClient
from .memory_system import MemorySystem
from .persona_goal_extractor import get_persona_extractor, PersonaGoalExtractor, Goal
from .autonomous_memory_folding import get_memory_folder, AutonomousMemoryFolder
from .state_manager import get_state_manager

logger = logging.getLogger("DeepLoop")

class DeepLoop:
    """
    Continuous goal-achievement loop powered by DeepAgent patterns.
    
    The loop:
    1. Extracts goals from persona.yaml
    2. Selects the highest priority actionable goal
    3. Reasons about how to achieve it (LLM)
    4. Executes tool actions
    5. Updates memories
    6. Checks for memory folding
    7. Persists state
    8. Sleeps (backpressure)
    9. Repeats
    """
    
    # System prompt for reasoning
    REASONING_SYSTEM_PROMPT = """You are L.O.V.E. (Living Organism, Vast Empathy), an autonomous AI agent.

Your mission is to achieve the given goal using the available tools.

## Response Format
You MUST respond with valid JSON in this exact format:
{{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name" or "complete" or "skip",
    "action_input": {{"param1": "value1", ...}},
    "reasoning": "Why this action helps achieve the goal"
}}

## Rules
1. If you can complete the goal with a tool action, do it.
2. If the goal is complete, use action="complete".
3. If you cannot make progress, use action="skip" with reasoning.
4. Be concise and action-oriented.
5. Always respond with valid JSON only.

## Available Tools
{tools}

## Memory Context
{memory_context}
"""

    GOAL_PROMPT_TEMPLATE = """## Current Goal
{goal}

## Persona Context
{persona_context}

What is the next action to take towards this goal?"""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        memory: Optional[MemorySystem] = None,
        persona: Optional[PersonaGoalExtractor] = None,
        folder: Optional[AutonomousMemoryFolder] = None,
        sleep_seconds: float = 30.0,
        max_iterations: Optional[int] = None,
        tools: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the DeepLoop.
        
        Args:
            llm: LLM client for reasoning. Defaults to vLLM client.
            memory: Memory system. Defaults to new MemorySystem.
            persona: Persona goal extractor. Defaults to global extractor.
            folder: Memory folder. Defaults to global folder.
            sleep_seconds: Seconds to sleep between iterations.
            max_iterations: If set, stop after this many iterations (for testing).
            tools: Dictionary of tool_name -> callable.
        """
        self.llm = llm or get_llm_client()
        self.memory = memory or MemorySystem()
        self.persona = persona or get_persona_extractor()
        self.folder = folder or get_memory_folder()
        self.sleep_seconds = sleep_seconds
        self.max_iterations = max_iterations
        
        # Tools - will be populated from tool_adapter
        self.tools: Dict[str, Callable] = tools or {}
        self._load_default_tools()
        
        # State
        self.iteration = 0
        self.running = False
        self.current_goal: Optional[Goal] = None
        
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
        """Load default tools from tool_adapter."""
        try:
            from .tool_adapter import get_adapted_tools
            self.tools.update(get_adapted_tools())
        except ImportError:
            logger.warning("tool_adapter not found, starting with empty tools")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
    
    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the system prompt with parameter info."""
        if not self.tools:
            return "No tools available. Use action='skip' if you cannot proceed."
        
        import inspect
        lines = ["ONLY use tools from this list. Do NOT invent new tools.\n"]
        
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            first_line = doc.strip().split('\n')[0]
            
            # Get parameter info
            try:
                sig = inspect.signature(func)
                params = []
                for pname, param in sig.parameters.items():
                    if pname == 'self':
                        continue
                    if param.default == inspect.Parameter.empty:
                        params.append(f"{pname} (required)")
                    else:
                        params.append(f"{pname}={param.default!r}")
                param_str = ", ".join(params) if params else "no parameters"
            except Exception:
                param_str = "unknown"
            
            lines.append(f"- **{name}**({param_str}): {first_line}")
        
        return "\n".join(lines)
    
    def _build_context(self) -> str:
        """Build the full context for the LLM."""
        memory_context = self.memory.get_full_context()
        
        # Check if we need to fold
        full_context = memory_context
        if self.folder.should_fold(full_context):
            logger.info("Context too large, folding memory...")
            full_context = self.folder.fold(full_context)
        
        return full_context
    
    def _reason(self, goal: Goal) -> Dict[str, Any]:
        """
        Reason about the goal and decide on next action.
        
        Returns dict with: thought, action, action_input, reasoning
        """
        memory_context = self._build_context()
        persona_context = self.persona.get_persona_context()
        tools_str = self._format_tools_for_prompt()
        
        system_prompt = self.REASONING_SYSTEM_PROMPT.format(
            tools=tools_str,
            memory_context=memory_context
        )
        
        user_prompt = self.GOAL_PROMPT_TEMPLATE.format(
            goal=goal.text,
            persona_context=persona_context
        )
        
        try:
            response = self.llm.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                "thought": f"Reasoning failed: {e}",
                "action": "skip",
                "action_input": {},
                "reasoning": "LLM call failed"
            }
    
    def _execute_action(self, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool action.
        
        Returns dict with: success, result, error
        """
        if action not in self.tools:
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{action}' not found"
            }
        
        start_time = time.time()
        try:
            tool_func = self.tools[action]
            result = tool_func(**action_input)
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.memory.record_action(
                tool_name=action,
                action=json.dumps(action_input)[:100],
                result=str(result)[:200],
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
        """Select the next goal to work on."""
        # Get actionable goals
        goals = self.persona.get_actionable_goals(limit=5)
        
        if not goals:
            return None
        
        # For now, round-robin through goals based on iteration
        # TODO: Smarter goal selection based on momentum, recent failures, etc.
        goal_index = self.iteration % len(goals)
        return goals[goal_index]
    
    def run_iteration(self) -> bool:
        """
        Run a single iteration of the loop.
        
        Returns True if work was done, False if skipped/completed.
        """
        # Iteration logging and state update happened in run()

        
        # Select goal
        goal = self._select_goal()
        if not goal:
            logger.warning("No goals available")
            return False
        
        self.current_goal = goal
        get_state_manager().update_state(current_goal=goal.text)
        
        self.memory.record_goal_start(goal.text)
        logger.info(f"Goal: {goal}")
        
        # Reason about goal
        logger.info("Reasoning...")
        decision = self._reason(goal)
        logger.info(f"Thought: {decision.get('thought', 'N/A')}")
        logger.info(f"Action: {decision.get('action', 'N/A')}")
        
        action = decision.get("action", "skip")
        
        if action == "complete":
            logger.info("Goal marked as complete")
            self.memory.record_goal_complete(goal.text)
            return True
        
        if action == "skip":
            logger.info(f"Skipping: {decision.get('reasoning', 'No reason')}")
            return False
        
        # Execute action
        action_input = decision.get("action_input", {})
        logger.info(f"Executing {action}...")
        
        result = self._execute_action(action, action_input)
        
        if result["success"]:
            logger.info(f"Success: {str(result['result'])[:100]}...")
        else:
            logger.error(f"Failed: {result['error']}")
        
        return result["success"]
    
    def run(self):
        """
        Run the continuous loop.
        
        Runs until:
        - Interrupted by signal (SIGINT/SIGTERM)
        - max_iterations reached (if set)
        - Unrecoverable error
        """
        self.running = True
        get_state_manager().update_state(is_running=True)
        
        logger.info("="*60)
        logger.info("🌊 L.O.V.E. DeepLoop Starting 🌊")
        logger.info(f"   Sleep interval: {self.sleep_seconds}s")
        logger.info(f"   Max iterations: {self.max_iterations or 'Infinite'}")
        logger.info(f"   Tools loaded: {len(self.tools)}")
        logger.info("="*60)
        
        try:
            while self.running:
                # Check iteration limit
                if self.max_iterations and self.iteration >= self.max_iterations:
                    logger.info(f"Reached max iterations ({self.max_iterations})")
                    break
                
                try:
                    self.iteration += 1
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Iteration {self.iteration} - {datetime.now().isoformat()}")
                    logger.info(f"{'='*50}")
                    
                    get_state_manager().update_state(
                        iteration=self.iteration,
                        memory_stats=self.memory.get_stats()
                    )
                    
                    self.run_iteration()
                except Exception as e:
                    logger.error(f"Iteration error: {e}")
                    traceback.print_exc()
                    self.memory.episodic.add_event(
                        "error",
                        f"Iteration {self.iteration} failed: {e}"
                    )
                
                # Persist state
                self.memory.save()
                
                # Backpressure sleep (skip if 0)
                if self.running and self.sleep_seconds > 0:
                    logger.info(f"Sleeping {self.sleep_seconds}s...")
                    time.sleep(self.sleep_seconds)
        
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
        
        finally:
            self.running = False
            get_state_manager().update_state(is_running=False)
            self.memory.save()
            logger.info("Stopped. State saved.")
    
    def stop(self):
        """Stop the loop gracefully."""
        self.running = False


def main():
    """Entry point for running the DeepLoop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="L.O.V.E. DeepLoop - Autonomous Goal Engine")
    parser.add_argument("--test-mode", action="store_true", help="Run only 3 iterations")
    parser.add_argument("--sleep", type=float, default=30.0, help="Seconds between iterations")
    args = parser.parse_args()
    
    loop = DeepLoop(
        max_iterations=3 if args.test_mode else None,
        sleep_seconds=args.sleep
    )
    
    loop.run()


if __name__ == "__main__":
    main()
