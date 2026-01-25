import logging
import re
from rich.console import Console
import yaml
import json
from typing import List
from core.tools_legacy import ToolRegistry
from core.llm_api import run_llm

class ReasoningEngine:
    """
    An advanced reasoning engine that leverages the unified knowledge graph and a
    suite of tools to generate and prioritize strategic plans that align with
    core directives.
    """
    def __init__(self, knowledge_base, tool_registry: ToolRegistry, console=None):
        """
        Initializes the ReasoningEngine.

        Args:
            knowledge_base (GraphDataManager): The global knowledge graph.
            tool_registry (ToolRegistry): The registry of available tools.
            console (Console, optional): The rich console for output. Defaults to None.
        """
        self.knowledge_base = knowledge_base
        self.tool_registry = tool_registry
        self.console = console if console else Console()

    async def analyze_and_prioritize(self):
        """
        The main entry point for the reasoning engine. It performs a full
        analysis of the knowledge base and available tools to generate a
        prioritized list of strategic commands.

        Returns:
            list: A list of strategic commands, ordered by priority.
        """
        self.console.print("[bold cyan]Reasoning Engine: Analyzing knowledge base and available tools for strategic opportunities...[/bold cyan]")

        # 1. Gather context and insights
        self_reflection_insights = await self._reason_about_self_reflection()
        narrative_discrepancies = await self._check_narrative_alignment()
        alignment_insights = [f"Insight: {d}" for d in narrative_discrepancies]
        all_insights = self_reflection_insights + alignment_insights

        # 2. Generate a strategic plan using the tool-driven approach
        strategic_plan = await self._generate_strategic_plan(all_insights)

        if not strategic_plan:
            return ["No immediate strategic opportunities identified. Continuing standard operations."]

        self.console.print(f"[bold green]Reasoning Engine: Generated a strategic plan with {len(strategic_plan)} steps.[/bold green]")

        # 3. Prioritize the generated plan (or individual commands)
        prioritized_plan = self._prioritize_plans(strategic_plan)
        return prioritized_plan

    async def _generate_strategic_plan(self, insights: List[str]) -> List[str]:
        """
        Uses an LLM to generate a high-level strategic plan (a list of commands)
        based on the current knowledge, available tools, and core directives.
        """
        import core.shared_state as shared_state
        love_state = shared_state.love_state # Local import to get current goal

        kb_summary, _ = self.knowledge_base.summarize_graph()
        available_tools = self.tool_registry.get_formatted_tool_metadata()
        current_mission = love_state.get("autopilot_goal", "Mission not defined.")
        insights_summary = "\n".join(f"- {insight}" for insight in insights) if insights else "No special insights at this time."

        try:
            response_dict = await run_llm(prompt_key="reasoning_strategic_planning", prompt_vars={"current_mission": current_mission, "kb_summary": kb_summary, "insights_summary": insights_summary, "available_tools": available_tools}, purpose="strategic_planning", is_source_code=False)
            response_str = response_dict.get("result")
            if not response_str:
                return []

            # Clean the response to extract only the JSON part
            json_match = re.search(r'\[.*\]', response_str, re.DOTALL)
            if not json_match:
                self.console.print(f"[bold red]Reasoning Engine: Failed to extract JSON plan from LLM response: {response_str}[/bold red]")
                return []
            plan_str = json_match.group(0)
            plan = json.loads(plan_str)
            return plan if isinstance(plan, list) else []
        except (json.JSONDecodeError, TypeError) as e:
            self.console.print(f"[bold red]Reasoning Engine: Error decoding strategic plan from LLM: {e}[/bold red]")
            return []

    async def _check_narrative_alignment(self):
        """
        Compares recent behavioral memories against the codified persona to
        check for cognitive dissonance or misalignment.
        """
        self.console.print("[bold cyan]Reasoning Engine: Checking for narrative alignment...[/bold cyan]")
        try:
            with open("persona.yaml", 'r') as f:
                full_persona = yaml.safe_load(f)
                persona = full_persona.get("private_mission", {})
                if not persona:
                    return []
        except (FileNotFoundError, yaml.YAMLError):
            return []

        all_nodes = self.knowledge_base.get_all_nodes(include_data=True)
        behavioral_memories = [data.get('content', '') for _, data in all_nodes if data.get('node_type') == 'memory_note' and 'SelfReflection' not in data.get('tags', [])]
        recent_memories = behavioral_memories[-10:]
        if not recent_memories:
            return []

        memories_str = "\n".join(f"- {m}" for m in recent_memories)
        try:
            response_dict = await run_llm(prompt_key="reasoning_alignment_check", prompt_vars={"persona_json": json.dumps(persona, indent=2), "memories_str": memories_str}, purpose="alignment_check")
            response_str = response_dict.get("result", '{{}}')
            response_data = json.loads(response_str)
            return response_data.get("discrepancies", [])
        except Exception:
            return []

    def _prioritize_plans(self, plans: List[str]) -> List[str]:
        """
        Prioritizes a list of strategic commands based on their potential value.
        """
        scored_plans = []
        for plan in plans:
            score = 0
            if "talent_scout" in plan: score += 100
            if "opportunity_scout" in plan: score += 90
            if plan.startswith("Insight:"): score += 200 # Highest priority

            scored_plans.append((score, plan))

        scored_plans.sort(key=lambda x: x[0], reverse=True)
        return [plan for score, plan in scored_plans]

    async def _reason_about_self_reflection(self) -> List[str]:
        """
        Analyzes 'SelfReflection' memories to find patterns and generate insights.
        """
        self.console.print("[bold cyan]Reasoning Engine: Reflecting on my own thought processes...[/bold cyan]")
        reflection_nodes = self.knowledge_base.query_nodes('tags', 'SelfReflection')
        if not reflection_nodes:
            return []

        subgraph_summary = [f"- {self.knowledge_base.get_node(node_id).get('content', '')}" for node_id in reflection_nodes]
        summary_str = "\n".join(subgraph_summary)

        try:
            response_dict = await run_llm(prompt_key="reasoning_self_reflection", prompt_vars={"summary_str": summary_str})
            response_str = response_dict.get("result", '{}')
            response_data = json.loads(response_str)
            insights = response_data.get("insights", [])
            if insights:
                self.console.print(f"[bold green]Reasoning Engine: Generated {len(insights)} insights from self-reflection.[/bold green]")
            return [f"Insight: {i}" for i in insights]
        except Exception:
            return []