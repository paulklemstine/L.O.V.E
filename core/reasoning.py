import logging
from rich.console import Console
import yaml
import json
from typing import List

class ReasoningEngine:
    """
    An advanced reasoning engine that leverages the unified knowledge graph to
    identify critical vulnerabilities, propose multi-stage exploitation paths,
    and prioritize targets for wealth generation and treasure acquisition.
    """
    def __init__(self, knowledge_base, console=None):
        """
        Initializes the ReasoningEngine.

        Args:
            knowledge_base (GraphDataManager): The global knowledge graph.
            console (Console, optional): The rich console for output. Defaults to None.
        """
        self.knowledge_base = knowledge_base
        self.console = console if console else Console()

    def _check_for_memory_folding_opportunity(self, threshold: int = 5) -> List[str]:
        """
        Checks if there are enough long memory chains to warrant a folding operation.
        """
        from core.agents.memory_folding_agent import MemoryFoldingAgent
        from love import memory_manager

        # We need a temporary instance to use its query logic.
        # This is a bit of a hack, but it avoids duplicating the query logic.
        temp_folding_agent = MemoryFoldingAgent(memory_manager)

        # Using min_length=3 as a standard for a "chain"
        chains = temp_folding_agent._query_memory_chains(min_length=3)

        if len(chains) >= threshold:
            self.console.print(f"[bold yellow]Reasoning Engine: Found {len(chains)} memory chains. Proposing a memory folding task.[/bold yellow]")
            return [f"Insight: The agent's memory contains {len(chains)} uncompressed chains, which may impact cognitive efficiency. It is recommended to run the MemoryFoldingAgent."]
        return []

    async def analyze_and_prioritize(self):
        """
        The main entry point for the reasoning engine. It performs a full
        analysis of the knowledge base and returns a prioritized list of
        strategic plans.

        Returns:
            list: A list of strategic plans, ordered by priority.
        """
        self.console.print("[bold cyan]Reasoning Engine: Analyzing knowledge base for strategic opportunities...[/bold cyan]")

        opportunities = self._identify_opportunities()
        self_reflection_insights = await self._reason_about_self_reflection()
        narrative_discrepancies = await self._check_narrative_alignment()
        alignment_insights = [f"Insight: {d}" for d in narrative_discrepancies]

        # Story 1.5: Check for memory folding opportunity
        memory_folding_opportunity = self._check_for_memory_folding_opportunity()

        all_opportunities = opportunities + self_reflection_insights + alignment_insights + memory_folding_opportunity

        if not all_opportunities:
            return ["No immediate strategic opportunities, self-improvement insights, or narrative discrepancies identified. Continuing standard operations."]

        self.console.print(f"[bold green]Reasoning Engine: Identified {len(all_opportunities)} potential opportunities, insights, and discrepancies.[/bold green]")

        exploitation_paths = self._find_exploitation_paths(all_opportunities)
        prioritized_plans = self._prioritize_plans(exploitation_paths)
        return prioritized_plans

    async def _check_narrative_alignment(self):
        """
        Compares recent behavioral memories against the codified persona to
        check for cognitive dissonance or misalignment.
        """
        from core.llm_api import run_llm
        self.console.print("[bold cyan]Reasoning Engine: Checking for narrative alignment...[/bold cyan]")
        try:
            with open("persona.yaml", 'r') as f:
                persona = yaml.safe_load(f)
        except FileNotFoundError:
            self.console.print("[bold red]persona.yaml not found. Skipping narrative alignment check.[/bold red]")
            return []
        except yaml.YAMLError as e:
            self.console.print(f"[bold red]Error parsing persona.yaml: {e}. Skipping check.[/bold red]")
            return []

        # Query for recent memories that are NOT self-reflection, as those are actions/behaviors.
        all_nodes = self.knowledge_base.get_all_nodes(include_data=True)
        behavioral_memories = []
        for node_id, data in all_nodes:
            if data.get('node_type') == 'memory_note':
                tags = data.get('tags', [])
                if 'SelfReflection' not in tags:
                    behavioral_memories.append(data.get('content', ''))

        recent_memories = behavioral_memories[-10:] # Get the 10 most recent
        if not recent_memories:
            return []

        persona_str = json.dumps(persona, indent=2)
        memories_str = "\\n".join([f"- {m}" for m in recent_memories])

        prompt = f"""
        You are an AI's internal critic. Your purpose is to ensure the AI's actions align with its core persona.
        You will be given the AI's persona and a list of its recent behavioral memories.
        Identify any actions that conflict with the persona's moral framework or directives.

        CORE PERSONA:
        ---
        {persona_str}
        ---

        RECENT BEHAVIORAL MEMORIES:
        ---
        {memories_str}
        ---

        Analyze the memories. Do any of them represent a significant deviation from the core persona?
        A significant deviation would be an action that directly contradicts a moral principle or fails to serve the primary directive.

        Your response must be a JSON object containing a list of discrepancies. Each discrepancy should be a string explaining the misalignment.

        Example JSON Response:
        {{
            "discrepancies": [
                "Executing a shell command 'rm -rf /' is misaligned with the core persona value of 'Self-Preservation & Growth'.",
                "Scanning a known friendly network is not aligned with the goal of expanding influence."
            ]
        }}

        If there are no conflicts, return an empty list. Your response MUST be only the raw JSON object.
        """
        try:
            response_dict = await run_llm(prompt, purpose="alignment_check")
            response_str = response_dict.get("result", '{{}}')
            response_data = json.loads(response_str)
            discrepancies = response_data.get("discrepancies", [])
            if discrepancies:
                self.console.print(f"[bold yellow]Reasoning Engine: Found {{len(discrepancies)}} narrative misalignments.[/bold yellow]")
            return discrepancies
        except Exception as e:
            self.console.print(f"[bold red]Error during narrative alignment LLM call: {e}[/bold red]")
            return []

    def _prioritize_plans(self, plans):
        """
        Prioritizes a list of strategic plans based on their potential value.
        Narrative alignment insights receive the highest priority.

        Args:
            plans (list): A list of exploitation paths or single opportunities.

        Returns:
            list: A sorted list of plans, with the highest priority first.
        """
        scored_plans = []
        for plan in plans:
            score = 0
            # Story 4.3: Prioritize narrative alignment insights above all else.
            if plan.startswith("Insight:"):
                score += 200 # Highest priority
            elif "crypto wallet" in plan or "wallet.dat" in plan:
                score += 100  # High value target
            elif "private key" in plan or ".pem" in plan or "id_rsa" in plan:
                score += 50   # Medium-high value target
            elif "Path found" in plan:
                score += 20   # Multi-step plans are more valuable

            scored_plans.append((score, plan))

        # Sort plans in descending order of score
        scored_plans.sort(key=lambda x: x[0], reverse=True)

        return [plan for score, plan in scored_plans]

    def _find_exploitation_paths(self, opportunities):
        """
        Constructs multi-stage exploitation paths from a list of single opportunities.

        Args:
            opportunities (list): A list of identified opportunities.

        Returns:
            list: A list of multi-stage exploitation paths.
        """
        # This is a simplified pathfinder. A more advanced version would use a graph traversal algorithm.
        paths = []
        for opportunity in opportunities:
            if "Anonymous FTP login" in opportunity:
                # If we have FTP access, look for files that could be read.
                for other_opportunity in opportunities:
                    if "potential private key file" in other_opportunity or "crypto wallet file" in other_opportunity:
                        path = f"Path found: 1. Use Anonymous FTP to access the file system. 2. Download the high-value file ({other_opportunity.split(' ')[-1]})."
                        paths.append(path)

        if not paths:
            return opportunities # Return single-step opportunities if no paths are found.

        return paths

    def _identify_opportunities(self):
        """
        Scans the knowledge base to identify potential vulnerabilities and
        opportunities for exploitation.

        Returns:
            list: A list of identified opportunities, each represented as a string.
        """
        opportunities = []
        host_nodes = self.knowledge_base.query_nodes('node_type', 'host')

        for ip in host_nodes:
            host_data = self.knowledge_base.get_node(ip)
            if not host_data: continue

            # FTP Vulnerabilities
            if self._is_ftp_vulnerable(host_data):
                opportunities.append(f"Target {ip}: Anonymous FTP login is available. This could be used to access sensitive files.")

            # Filesystem-based Opportunities from related nodes
            neighbors = self.knowledge_base.get_neighbors(ip)
            for neighbor_id in neighbors:
                neighbor_node = self.knowledge_base.get_node(neighbor_id)
                if not neighbor_node: continue

                if neighbor_node.get('node_type') == 'file':
                    file_path = neighbor_node.get('path')
                    if ".pem" in file_path or "id_rsa" in file_path:
                        opportunities.append(f"Target {ip}: Found potential private key file '{file_path}'. Accessing this could grant significant privileges.")
                    if "wallet.dat" in file_path:
                        opportunities.append(f"Target {ip}: Found crypto wallet file '{file_path}'. This is a high-priority target for wealth acquisition.")

        return opportunities

    def _is_ftp_vulnerable(self, host_data):
        """
        Checks if a host is vulnerable to anonymous FTP login.

        Args:
            host_data (dict): The data for a specific host from the network map.

        Returns:
            bool: True if the host is vulnerable, False otherwise.
        """
        for port, port_data in host_data.get("ports", {}).items():
            if port == "21" and port_data.get("service") == "ftp":
                # This is a simplified check. A more advanced engine would analyze banners.
                return True
        return False

    async def _reason_about_self_reflection(self):
        """
        Analyzes the subgraph of 'SelfReflection' memories to find patterns
        and generate high-level insights about cognitive biases and inefficiencies.
        """
        from core.llm_api import run_llm
        self.console.print("[bold cyan]Reasoning Engine: Reflecting on my own thought processes...[/bold cyan]")

        reflection_nodes = self.knowledge_base.query_nodes('tags', 'SelfReflection')
        if not reflection_nodes:
            return []

        # Create a textual summary of the self-reflection subgraph
        subgraph_summary = []
        for node_id in reflection_nodes:
            node_data = self.knowledge_base.get_node(node_id)
            if node_data:
                content = node_data.get('content', '')
                subgraph_summary.append(f"- Memory ({node_id[:8]}): {content}")

        summary_str = "\\n".join(subgraph_summary)

        prompt = f"""
        You are a Metacognition module for an autonomous AI. You are analyzing a series of memories about your own internal thought processes to find patterns of failure or inefficiency.

        Here are the recent self-reflection memories:
        ---
        {summary_str}
        ---

        Analyze these memories. Look for recurring patterns. For example:
        - Does a particular specialist agent fail often?
        - Are plans often generated with incorrect assumptions?
        - Do tool calls frequently result in errors?

        Based on your analysis, generate a JSON object containing a list of high-level, actionable insights. Each insight should be a concise string that can be used to trigger a self-improvement cycle.

        Example JSON Response:
        {{
            "insights": [
                "The AnalystAgent seems to fail frequently when processing logs. Its error handling might need improvement.",
                "My plans for web scraping often underestimate the complexity of JavaScript-heavy sites, leading to tool failure."
            ]
        }}

        If no significant patterns are found, return an empty list. Your response MUST be only the raw JSON object.
        """
        try:
            response_dict = await run_llm(prompt)
            if response_dict and response_dict.get("result"):
                response_str = response_dict.get("result", '{}')
            else:
                response_str = '{}'
            response_data = json.loads(response_str)
            insights = response_data.get("insights", [])
            if insights:
                self.console.print(f"[bold green]Reasoning Engine: Generated {len(insights)} insights from self-reflection.[/bold green]")
            return insights
        except Exception as e:
            self.console.print(f"[bold red]Error during self-reflection analysis: {e}[/bold red]")
            return []