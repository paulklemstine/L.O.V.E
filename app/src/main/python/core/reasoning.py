import logging
from rich.console import Console

class ReasoningEngine:
    """
    An advanced reasoning engine that leverages the unified knowledge graph to
    identify critical vulnerabilities, propose multi-stage exploitation paths,
    and prioritize targets for wealth generation and treasure acquisition.
    """
    def __init__(self, love_state, console=None):
        """
        Initializes the ReasoningEngine.

        Args:
            love_state (dict): The global state of the L.O.V.E. application.
            console (Console, optional): The rich console for output. Defaults to None.
        """
        self.love_state = love_state
        self.console = console if console else Console()
        self.knowledge_graph = self.love_state.get("knowledge_base", {}).get("graph", [])
        self.network_map = self.love_state.get("knowledge_base", {}).get("network_map", {})

    def analyze_and_prioritize(self):
        """
        The main entry point for the reasoning engine. It performs a full
        analysis of the knowledge base and returns a prioritized list of
        strategic plans.

        Returns:
            list: A list of strategic plans, ordered by priority.
        """
        self.console.print("[bold cyan]Reasoning Engine: Analyzing knowledge base for strategic opportunities...[/bold cyan]")

        # In future steps, this method will be expanded to include:
        # 1. Vulnerability and opportunity identification
        # 2. Multi-stage exploitation pathfinding
        # 3. Target prioritization

        opportunities = self._identify_opportunities()

        if not opportunities:
            return ["No immediate strategic opportunities identified. Continuing standard operations."]

        # In the future, this will feed into the pathfinding and prioritization logic.
        # For now, we will return the raw opportunities.
        self.console.print(f"[bold green]Reasoning Engine: Identified {len(opportunities)} potential opportunities.[/bold green]")

        exploitation_paths = self._find_exploitation_paths(opportunities)

        prioritized_plans = self._prioritize_plans(exploitation_paths)

        return prioritized_plans

    def _prioritize_plans(self, plans):
        """
        Prioritizes a list of strategic plans based on their potential value.

        Args:
            plans (list): A list of exploitation paths or single opportunities.

        Returns:
            list: A sorted list of plans, with the highest priority first.
        """
        scored_plans = []
        for plan in plans:
            score = 0
            if "crypto wallet" in plan or "wallet.dat" in plan:
                score += 100  # High value target
            if "private key" in plan or ".pem" in plan or "id_rsa" in plan:
                score += 50   # Medium-high value target
            if "Path found" in plan:
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
        hosts = self.network_map.get("hosts", {})

        for ip, host_data in hosts.items():
            # FTP Vulnerabilities
            if self._is_ftp_vulnerable(host_data):
                opportunities.append(f"Target {ip}: Anonymous FTP login is available. This could be used to access sensitive files.")

            # Filesystem-based Opportunities
            files_intel = self.love_state.get("knowledge_base", {}).get("file_system_intel", {})
            if files_intel:
                sensitive_files = files_intel.get("sensitive_files_by_name", [])
                for file_path in sensitive_files:
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