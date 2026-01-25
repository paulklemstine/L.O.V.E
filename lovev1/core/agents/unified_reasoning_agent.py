from typing import Dict, List, Optional, Any
import asyncio
import json

from core.agents.memory_folding_agent import MemoryFoldingAgent
from core.llm_api import execute_reasoning_task
from core.memory.memory_manager import MemoryManager


class UnifiedReasoningAgent:
    """
    An agent designed for open-ended reasoning, equipped with a brain-inspired memory schema.
    It manages its own episodic, working, and tool memory in a unified reasoning process.
    
    Supports dynamic tool context injection via the tools and tool_registry parameters.
    """
    def __init__(
        self, 
        memory_manager: MemoryManager,
        tools: Optional[List[Any]] = None,
        tool_registry: Optional['ToolRegistry'] = None
    ):
        self.memory_manager = memory_manager
        self.internal_history = []
        self.history_threshold = 5
        self.tools = tools or []
        self.tool_registry = tool_registry
    
    def _build_available_tools_section(self) -> str:
        """
        Builds the ## Available Tools section for system prompts.
        
        Uses OpenAI/Gemini function calling schema format for tool definitions.
        Prefers tool_registry schemas if available, falls back to tools list.
        
        Returns:
            Formatted string with tool definitions, or empty string if no tools.
        """
        if not self.tool_registry and not self.tools:
            return ""
        
        output = "\n## Available Tools\n\n"
        output += "You have access to the following tools. To use a tool, respond with a JSON block containing 'tool' and 'arguments' keys.\n\n"
        
        # Prefer registry schemas if available
        if self.tool_registry:
            try:
                schemas = self.tool_registry.get_all_tool_schemas()
                for schema in schemas:
                    output += f"### {schema['name']}\n"
                    output += f"**Description:** {schema.get('description', 'No description')}\n"
                    params = schema.get('parameters', {})
                    if params.get('properties'):
                        output += "**Parameters (JSON Schema):**\n```json\n"
                        output += json.dumps(params, indent=2)
                        output += "\n```\n"
                    else:
                        output += "**Parameters:** None\n"
                    output += "\n"
            except Exception as e:
                output += f"(Error loading tool schemas: {e})\n"
        elif self.tools:
            for tool in self.tools:
                tool_name = getattr(tool, 'name', str(tool))
                tool_desc = getattr(tool, 'description', 'No description')
                output += f"### {tool_name}\n"
                output += f"**Description:** {tool_desc}\n\n"
        
        return output

    async def execute_task(self, task_details: Dict) -> str:
        """
        The main entry point for the agent's reasoning loop.
        """
        goal = task_details.get("goal", "No goal specified.")
        print(f"--- UnifiedReasoningAgent starting task: {goal} ---")

        self.internal_history.append(f"Initial Goal: {goal}")

        # Reasoning loop
        for _ in range(10): # Limit to 10 steps to prevent infinite loops
            # 1. Retrieve relevant memories
            relevant_memories = self.memory_manager.retrieve_relevant_folded_memories(goal, top_k=3)
            print(f"Retrieved {len(relevant_memories)} relevant folded memories.")

            # 2. Build the reasoning prompt
            from core.prompt_registry import PromptRegistry
            registry = PromptRegistry()
            
            # Pull a community-optimized reasoning prompt if available, or fall back to local/hardcoded
            remote_prompt = registry.get_hub_prompt("hwchase17/react")
            
            # If we got a valid remote prompt, we might want to adapt it or just use it as a base.
            # For this integration, let's incorporate it into our context.
            
            history_str = "\n".join(self.internal_history)
            
            # Build available tools section for dynamic context injection
            tools_section = self._build_available_tools_section()
            
            prompt = f"""You are a Unified Reasoning Agent. Your goal is to solve complex, open-ended problems.

**Community Wisdom (Hub Prompt):**
{remote_prompt}
{tools_section}
**Your Goal:**
{goal}

**Relevant Past Experiences (Summaries):**
{relevant_memories}

**Current Task History:**
{history_str}

Based on all of this context, what is the very next, single, atomic action you should take?
Your response should be ONLY the thought process and the command to execute.
"""
            # 3. Execute a unified reasoning step
            result = await execute_reasoning_task(prompt)
            print(f"Unified reasoning step result: {result}")
            self.internal_history.append(result)

            # Check for completion
            if "task complete" in result.lower() or "goal achieved" in result.lower():
                print("--- UnifiedReasoningAgent determined task is complete. ---")
                break

            # 4. Check if history needs to be folded
            if len(self.internal_history) >= self.history_threshold:
                print(f"--- Internal history threshold reached ({len(self.internal_history)}). Triggering memory folding. ---")
                # Add the current history to the main memory graph as a chain
                for memory in self.internal_history:
                    await self.memory_manager.add_episode(memory, tags=["UnifiedReasoningCycle"])

                # Now, run the folding agent
                memory_folder = MemoryFoldingAgent(self.memory_manager)
                folding_result = await memory_folder.execute_task({"min_length": self.history_threshold})
                print(f"Memory folding result: {folding_result}")

                # Clear the internal history
                self.internal_history = ["History has been folded into a summary."]


        final_answer = self.internal_history[-1] if self.internal_history else "No result was generated."
        print(f"--- UnifiedReasoningAgent finished task. Final Answer: {final_answer} ---")
        return final_answer

