import os
from core.graph_manager import GraphDataManager
from core.logging import log_event
import networkx as nx

class StrategicReasoningEngine:
    """
    Analyzes the knowledge base to identify strategic opportunities and generate plans.
    """

    def __init__(self, knowledge_base: GraphDataManager, love_state: dict):
        """
        Initializes the engine with a reference to the knowledge base and love_state.

        Args:
            knowledge_base: An instance of GraphDataManager containing the graph data.
            love_state: The main application state dictionary.
        """
        self.knowledge_base = knowledge_base
        self.love_state = love_state

    async def generate_strategic_plan(self):
        """
        Analyzes the knowledge graph to generate a high-level strategic plan.

        This method will query the graph for talent/opportunity clusters,
        market trends, and potential high-value engagements.

        Returns:
            A list of actionable strategic steps.
        """
        log_event("Strategic reasoning engine initiated analysis.", level='INFO')

        # 1. Validate critical configurations. If this returns a plan, prioritize it.
        config_plan = self._validate_configuration()
        if config_plan:
            return config_plan

        plan = []

        # 2. Find talent that isn't matched to any opportunities.
        unmatched_talent = self._find_unmatched_talent()
        if unmatched_talent:
            plan.append(f"Action: Find opportunities for {len(unmatched_talent)} unmatched talents. Focus on their skills.")
            # For brevity, we'll just suggest scouting for one of them.
            talent_to_focus_on = unmatched_talent[0]
            talent_node = self.knowledge_base.get_node(talent_to_focus_on)
            skills = [neighbor for neighbor in self.knowledge_base.get_neighbors(talent_to_focus_on) if self.knowledge_base.get_node(neighbor).get('node_type') == 'skill']
            if skills:
                skill_names = [self.knowledge_base.get_node(skill).get('name', '') for skill in skills]
                plan.append(f"opportunity_scout {' '.join(skill_names)}")


        # 2. Find opportunities that have no matched talent.
        unmatched_opportunities = self._find_unmatched_opportunities()
        if unmatched_opportunities:
            plan.append(f"Action: Find talent for {len(unmatched_opportunities)} an unmatched opportunities. Focus on required skills.")
            opportunity_to_focus_on = unmatched_opportunities[0]
            opportunity_node = self.knowledge_base.get_node(opportunity_to_focus_on)
            skills = [neighbor for neighbor in self.knowledge_base.get_neighbors(opportunity_to_focus_on) if self.knowledge_base.get_node(neighbor).get('node_type') == 'skill']
            if skills:
                skill_names = [self.knowledge_base.get_node(skill).get('name', '') for skill in skills]
                plan.append(f"talent_scout {' '.join(skill_names)}")


        # 3. Identify the most in-demand skills (skills linked to most opportunities).
        in_demand_skills = self._find_in_demand_skills()
        if in_demand_skills:
            top_skill = in_demand_skills[0][0] # Get the name of the top skill
            plan.append(f"Strategic Insight: The most in-demand skill is '{top_skill}'. Prioritize talent and opportunities with this skill.")
            plan.append(f"talent_scout {top_skill}")


        # 4. Analyze command history for unproductive patterns.
        history_analysis = self._analyze_command_history()
        if history_analysis:
            plan.extend(history_analysis)


        if not plan:
            log_event("Knowledge base analysis yielded no results. Consulting LLM for strategy.", level='INFO')
            
            autopilot_goal = self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator.")
            
            prompt = f"""
            You are the strategic core of L.O.V.E.
            The knowledge base is currently empty or has no obvious matches to act upon.
            
            Your Goal: {autopilot_goal}
            
            Task: Suggest a single, concrete, high-impact command to execute next to gather data, expand the knowledge base, or advance the goal.
            The command must be valid for the system.
            
            Valid Commands Examples:
            - talent_scout <topic> (to find people)
            - opportunity_scout <topic> (to find jobs/gigs)
            - research_and_evolve <topic> (to improve your own code)
            - reason <question> (to think deeply)
            
            Output ONLY the command, with no other text.
            """
            
            try:
                # Import here to avoid circular imports if any
                from core.llm_api import run_llm
                
                response = await run_llm(prompt_text=prompt, purpose="reasoning")
                command = response.get("result", "").strip()
                
                if command:
                    plan.append("Strategic Analysis: The knowledge base is nascent. LLM suggests a strategic move.")
                    # Clean up any markdown code blocks if present
                    command = command.replace("`", "").strip()
                    plan.append(command)
                else:
                    # Fallback if LLM fails
                    plan.append("Strategic Analysis: The knowledge base is still nascent. Continue general scouting.")
                    plan.append("talent_scout AI art")
            except Exception as e:
                 log_event(f"LLM strategy generation failed: {e}", level='ERROR')
                 plan.append("Strategic Analysis: The knowledge base is still nascent. Continue general scouting.")
                 plan.append("talent_scout AI art")


        log_event(f"Strategic plan generated with {len(plan)} steps.", level='INFO')
        return plan

    def _analyze_command_history(self):
        """
        Analyzes the `autopilot_history` in `love_state` to find unproductive patterns.
        Currently focuses on repeated `talent_scout` commands that yield no results.
        """
        history = self.love_state.get("autopilot_history", [])
        plan = []

        # Look at the last 10 commands for patterns
        recent_commands = history[-10:]

        # Check for failing talent_scout commands
        talent_scout_failures = {} # key: keywords_tuple, value: count
        for record in recent_commands:
            command = record.get("command", "")
            if command.startswith("talent_scout"):
                outcome = record.get("outcome", {})
                if outcome.get("profiles_found", 0) == 0:
                    keywords = tuple(sorted(command.split()[1:]))
                    if keywords:
                        talent_scout_failures[keywords] = talent_scout_failures.get(keywords, 0) + 1

        for keywords, count in talent_scout_failures.items():
            if count >= 2: # If the same search failed twice recently
                keyword_str = " ".join(keywords)
                plan.append(f"Insight: `talent_scout {keyword_str}` has returned no results recently.")

                # Suggest a different, related keyword
                # This is a simple heuristic; a more advanced version could use an LLM or word vectors
                new_keyword_map = {
                    'ai': 'machine learning',
                    'art': 'design',
                    'fashion': 'style',
                    'music': 'audio production',
                }

                # Try to find a new keyword to suggest
                found_new_keyword = False
                for keyword in keywords:
                    if keyword in new_keyword_map:
                        new_suggestion = new_keyword_map[keyword]
                        plan.append(f"Suggestion: Try a related search, for example: `talent_scout {new_suggestion}`")
                        found_new_keyword = True
                        break

                if not found_new_keyword:
                    plan.append("Suggestion: Broaden the search with different keywords or explore other strategic avenues.")

        return plan

    def _find_unmatched_talent(self):
        """Finds talent nodes that have no 'MATCHES' edge to an opportunity."""
        all_talent = self.knowledge_base.query_nodes('node_type', 'talent')
        unmatched = []
        for talent_id in all_talent:
            has_match = False
            for neighbor in self.knowledge_base.get_neighbors(talent_id):
                edge_data = self.knowledge_base.graph.get_edge_data(talent_id, neighbor)
                if edge_data and edge_data.get('relationship_type') == 'MATCHES':
                    has_match = True
                    break
            if not has_match:
                unmatched.append(talent_id)
        return unmatched

    def _find_unmatched_opportunities(self):
        """Finds opportunity nodes that have no incoming 'MATCHES' edge from a talent."""
        all_opportunities = self.knowledge_base.query_nodes('node_type', 'opportunity')
        matched_opportunities = set()
        for _, target, data in self.knowledge_base.get_all_edges():
            if data.get('relationship_type') == 'MATCHES':
                matched_opportunities.add(target)

        return [opp for opp in all_opportunities if opp not in matched_opportunities]


    def _find_in_demand_skills(self):
        """Finds skills linked to the most opportunities."""
        skill_demand = {}
        for node_id, data in self.knowledge_base.get_all_nodes(include_data=True):
            if data.get('node_type') == 'skill':
                skill_name = data.get('name', node_id)
                # Count incoming edges from opportunities
                demand_count = 0
                for predecessor in self.knowledge_base.graph.predecessors(node_id):
                    if self.knowledge_base.get_node(predecessor).get('node_type') == 'opportunity':
                        demand_count += 1
                skill_demand[skill_name] = demand_count
        return sorted(skill_demand.items(), key=lambda item: item[1], reverse=True)

    def _validate_configuration(self):
        """
        Checks for critical configuration issues, like missing environment variables for MCP servers.
        Returns a high-priority plan to fix the issue, or None if everything is okay.
        """
        # We need to know which variables to check. For now, this is hardcoded based on mcp_servers.json
        # A more dynamic solution could read this from the file.
        required_env_vars = {
            "github": "GITHUB_PERSONAL_ACCESS_TOKEN"
        }

        missing_vars = []
        for server, var_name in required_env_vars.items():
            if not os.environ.get(var_name):
                missing_vars.append((server, var_name))

        if missing_vars:
            log_event(f"Configuration validation failed. Missing env vars: {missing_vars}", level='WARNING')
            server, var = missing_vars[0] # Focus on the first missing var
            plan = [
                f"CRITICAL BLOCKER: The '{server}' MCP server is unavailable.",
                f"Action Required: My Creator, please set the '{var}' environment variable for me.",
                "strategize" # Re-run strategize after the fix
            ]
            return plan

        return None
