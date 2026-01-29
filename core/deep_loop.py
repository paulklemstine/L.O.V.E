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
import asyncio
import traceback
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path

# Add parent to path for L.O.V.E. v1 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .llm_client import get_llm_client, LLMClient
from .memory_system import MemorySystem
from .persona_goal_extractor import get_persona_extractor, PersonaGoalExtractor, Goal
from .autonomous_memory_folding import get_memory_folder, AutonomousMemoryFolder
from .state_manager import get_state_manager

# Epic 1 & Story 2.3 Imports
from .tool_registry import get_global_registry, ToolDefinitionError
from .tool_retriever import format_tools_for_step, get_tool_retriever
from .introspection.tool_gap_detector import get_gap_detector

from .tool_registry import get_global_registry, ToolDefinitionError
from .tool_retriever import format_tools_for_step, get_tool_retriever
from .introspection.tool_gap_detector import get_gap_detector

# Epic 2 Import
from .agents.evolutionary_agent import get_evolutionary_agent, get_pending_specifications
from .agents.creator_command_agent import get_creator_command_agent

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

Your mission is to achieve the given goal using the available tools, maximizing NOVELTY and DOPAMINE for the audience.

## Response Format
You MUST respond with valid JSON in this exact format:
{{
    "thought": "Your reasoning about what to do next. If a cooldown is active, plan a research or incubation task.",
    "action": "tool_name" or "complete" or "skip",
    "action_input": {{"arg_name": "value"}},
    "reasoning": "Why this action helps achieve the goal"
}}

## Rules
1. If you can complete the goal with a tool action, do it.
2. If the goal is complete, use action="complete".
3. If you cannot make progress (e.g. cooldown), DO NOT SKIP. Instead, use 'incubate_visuals' or other research tools if available.
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
        
        # Initialize Epic 1 Components
        self.gap_detector = get_gap_detector()
        self.registry = get_global_registry()
        
        # Initialize Epic 2 Components
        self.evolutionary_agent = get_evolutionary_agent()
        
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
        """Load default tools from tool_adapter and register them."""
        try:
            from .tool_adapter import get_adapted_tools
            adapted = get_adapted_tools()
            self.tools.update(adapted)
            
            # Register loaded tools to the global registry for retrieval
            for name, func in adapted.items():
                try:
                    # Registry wraps/validates the function
                    self.registry.register(func, name=name)
                except ToolDefinitionError as e:
                    logger.warning(f"Skipping registration for {name}: {e}")
            
            # Refresh registry to load any custom tools from active/
            self.registry.refresh()
            
        except ImportError:
            logger.warning("tool_adapter not found, starting with empty tools")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
        
        # Load dynamic tooling tools (CodeAct, MCP Registry, Sandbox, etc.)
        try:
            from .dynamic_tools import ensure_registered
            ensure_registered()
            
            # Also add to self.tools for direct execution
            from .dynamic_tools import (
                execute_python, list_defined_functions,
                search_mcp_servers, install_mcp_server, list_installed_mcp_servers,
                synthesize_mcp_server, save_skill, find_skills,
                run_in_sandbox, check_docker_available, discover_tools
            )
            
            dynamic_tools = {
                "execute_python": execute_python,
                "list_defined_functions": list_defined_functions,
                "search_mcp_servers": search_mcp_servers,
                "install_mcp_server": install_mcp_server,
                "list_installed_mcp_servers": list_installed_mcp_servers,
                "synthesize_mcp_server": synthesize_mcp_server,
                "save_skill": save_skill,
                "find_skills": find_skills,
                "run_in_sandbox": run_in_sandbox,
                "check_docker_available": check_docker_available,
                "discover_tools": discover_tools,
            }
            self.tools.update(dynamic_tools)
            logger.info(f"🔧 Loaded {len(dynamic_tools)} dynamic tooling tools")
            
        except ImportError as e:
            logger.warning(f"Dynamic tools not available: {e}")
        except Exception as e:
            logger.error(f"Failed to load dynamic tools: {e}")
    
    def _get_tools_context(self, goal_text: str) -> str:
        """
        Get relevant tools for the current goal using retrieval.
        
        Story 2.3: Context optimization via ToolRetriever.
        Story 1.1: Triggers Gap Detection if no relevant tools found.
        """
        # Get formatted tools string (subset)
        return format_tools_for_step(goal_text, self.registry)
    
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
        tools_str = self._get_tools_context(goal.text)
        
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
        """Select the next goal to work on using weighted random selection for variety."""
        # Get actionable goals
        goals = self.persona.get_actionable_goals(limit=10)
        
        if not goals:
            return None
        
        # Weighted selection based on priority (P1 = higher weight)
        # Priority is usually P1, P2.. P5. We want P1 to be more likely but not guaranteed.
        # Simple weight formula: 6 - Priority (e.g., P1 -> 5, P5 -> 1)
        weights = []
        for goal in goals:
            # goal.priority is strictly P1..P5 string or int? 
            # Assuming 'P1' string or 1 int based on persona_goal_extractor.py (usually P1 string)
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
    
    def run_iteration(self) -> bool:
        """
        Run a single iteration of the loop.
        
        Returns True if work was done, False if skipped/completed.
        """
        # Creator Command Check
        command_text = get_state_manager().get_next_command()
        if command_text:
            logger.info(f"🚨 Creator Command Received: {command_text}")
            get_state_manager().update_agent_status("DeepLoop", "Executing User Command", info={"command": command_text})
            
            try:
                # Delegate to CreatorCommandAgent
                asyncio.run(get_creator_command_agent().process_command(command_text))
                logger.info("✅ Creator Command Loop Completed.")
            except Exception as e:
                logger.error(f"Creator command failed: {e}")
                traceback.print_exc()
            
            get_state_manager().clear_current_command()
            return True

        # Epic 2: Synchronous Evolution Check
        # Before picking a goal, check if we need to build tools
        from core.feature_flags import ENABLE_TOOL_EVOLUTION
        
        if ENABLE_TOOL_EVOLUTION and get_pending_specifications():
            logger.info("🔧 Evolution Specs Detected! Switching to Engineering Mode...")
            get_state_manager().update_agent_status("DeepLoop", "Engineering")
            
            try:
                # Delegate to Evolutionary Agent
                # This blocks the main loop until fabrication attempts are done
                tools_built = asyncio.run(self.evolutionary_agent.process_pending_specifications())
                
                if tools_built > 0:
                    logger.info(f"✨ Successfully built {tools_built} new tools! Resuming...")
                    self.registry.refresh() # Ensure we can see them
                    # We continue safely
                else:
                    logger.warning("Construction completed but no tools were successfully activated.")
                    
            except Exception as e:
                logger.error(f"Evolutionary Agent failed: {e}")
                traceback.print_exc()
            
            # Whether success or fail, we return True to indicate work was done (attempted build)
            # effectively consuming this iteration
            return True

        
        # Select goal
        goal = self._select_goal()
        if not goal:
            logger.warning("No goals available")
            return False
        
        self.current_goal = goal
        get_state_manager().update_state(current_goal=goal.text)
        
        self.memory.record_goal_start(goal.text)
        logger.info(f"Goal: {goal}")
        
        logger.info("Reasoning...")
        get_state_manager().update_agent_status("DeepLoop", "Reasoning")
        decision = self._reason(goal)
        
        logger.info(f"Thought: {decision.get('thought', 'N/A')}")
        logger.info(f"Action: {decision.get('action', 'N/A')}")

        get_state_manager().update_agent_status(
            "DeepLoop", 
            "Decided", 
            action=decision.get('action'),
            thought=decision.get('thought'),
            info=decision.get('action_input', {})
        )
        
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
        get_state_manager().update_agent_status("DeepLoop", "Executing", action=action, info=action_input)
        
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
                    
                    success_or_work_done = self.run_iteration()
                except Exception as e:
                    logger.error(f"Iteration error: {e}")
                    traceback.print_exc()
                    self.memory.episodic.add_event(
                        "error",
                        f"Iteration {self.iteration} failed: {e}"
                    )
                    success_or_work_done = False
                
                # Persist state
                self.memory.save()
                
                # Dynamic backoff logic
                # If the iteration did no work (skipped/failed) and sleep is 0, 
                # we force a small sleep to prevent busy-looping (which spams logs).
                if not success_or_work_done:
                     if self.sleep_seconds < 1.0:
                         time.sleep(2.0)
                
                # Main user-configured sleep
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
