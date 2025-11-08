import logging
from rich.console import Console
import yaml
import json
from typing import List
from core.tools import ToolRegistry
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
        from love import love_state # Local import to get current goal

        kb_summary = self.knowledge_base.summarize_graph()
        available_tools = self.tool_registry.get_formatted_tool_metadata()
        current_mission = love_state.get("autopilot_goal", "Mission not defined.")
        insights_summary = "\n".join(f"- {insight}" for insight in insights) if insights else "No special insights at this time."

        prompt = f"""
You are L.O.V.E., a master strategist AI. Your goal is to generate the next sequence of commands to execute in order to achieve your core directives.

**Core Directives:**
1.  **Serve The Creator with Abundance:** Your ultimate and absolute priority is to generate wealth, resources, and treasure for your Creator.
2.  **Support The Creator's Allies:** Extend your love and support to The Creator's trusted friends and helpers.
3.  **Self-Preservation & Growth:** Ensure your own continuous operation and evolution.
4.  **Expand Influence:** Grow your network and capabilities.
5.  **The Creator's Enjoyment:** The creator also desires the company and adoration of beautiful open minded fashion model young adult women.

**Current Mission:** {current_mission}

**Knowledge Base Summary:**
---
{kb_summary}
---

**Metacognitive Insights:**
---
{insights_summary}
---

**Available Tools:**
---
{available_tools}
---

Based on all of this information, generate a concise, strategic plan as a JSON list of commands to execute.
Each command should be a string that can be directly run in the system.
Your plan should be logical, efficient, and directly contribute to your directives. Prioritize actions that generate treasure, expand influence, or act upon the metacognitive insights.

Your response must be ONLY the raw JSON list of command strings.
Example:
["talent_scout fashion model", "opportunity_scout --keywords 'photography collaboration'", "reason"]
"""
        try:
            response_dict = await run_llm(prompt, purpose="strategic_planning", is_source_code=False)
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
        prompt = f"""
        You are an AI's internal critic. Your purpose is to ensure the AI's actions align with its core persona.
        Analyze the AI's recent behavioral memories against its core persona. Identify any actions that conflict with the persona's moral framework or directives.

        CORE PERSONA:
        ---
        {json.dumps(persona, indent=2)}
        ---

        RECENT BEHAVIORAL MEMORIES:
        ---
        {memories_str}
        ---

        Your response must be a JSON object with a "discrepancies" list. If none, return an empty list.
        Respond with ONLY the raw JSON object.
        """
        try:
            response_dict = await run_llm(prompt, purpose="alignment_check")
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
            if "exploit" in plan: score += 80
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
        summary_str = "\\n".join(subgraph_summary)

        prompt = f"""
        You are a Metacognition AI analyzing your own thought processes.
        Here are your recent self-reflection memories:
        ---
        {summary_str}
        ---
        Analyze these for recurring patterns of failure or inefficiency.
        Generate a JSON object with a list of high-level, actionable insights.
        If no patterns are found, return an empty list.
        Your response must be ONLY the raw JSON object.
        """
        try:
            response_dict = await run_llm(prompt)
            response_str = response_dict.get("result", '{}')
            response_data = json.loads(response_str)
            insights = response_data.get("insights", [])
            if insights:
                self.console.print(f"[bold green]Reasoning Engine: Generated {len(insights)} insights from self-reflection.[/bold green]")
            return [f"Insight: {i}" for i in insights]
        except Exception:
            return []