import os
import json
from core.graph_manager import GraphDataManager
from core.logging import log_event
import time
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


    def _load_persona(self):
        """Loads the persona configuration."""
        # Try to find persona.yaml relative to this file
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persona_path = os.path.join(base_dir, "persona.yaml")
            if os.path.exists(persona_path):
                import yaml
                with open(persona_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            log_event(f"Error loading persona in StrategicReasoningEngine: {e}", level='ERROR')
        return {}

    async def generate_strategic_plan(self, reflexion_context: str = None):
        """
        Analyzes the knowledge graph and persona priorities to generate a strategic plan.
        """
        log_event("Strategic reasoning engine initiated analysis.", level='INFO')

        # 1. Validate critical configurations.
        config_plan = self._validate_configuration()
        if config_plan:
            return config_plan

        plan = []
        if reflexion_context:
            log_event(f"Applying Reflexion Context: {reflexion_context}", level='INFO')
            plan.append(f"Reflexion Adjustment: {reflexion_context}")

        # 2. Load Goals from Persona
        persona = self._load_persona()
        priorities = persona.get('private_mission', {}).get('core_priorities', {})
        
        if not priorities:
             # Fallback if persona not loaded
             log_event("Persona priorities not found. Using default logic.", level='WARNING')
             priorities = {1: "Evolve technically", 2: "Make Creator wealthy"}

        # 3. Dynamic Goal Handling
        for rank, goal_desc in sorted(priorities.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 99):
            log_event(f"Processing Priority {rank}: {goal_desc}", level='DEBUG')
            
            goal_lower = goal_desc.lower()
            
            if "wealth" in goal_lower or "money" in goal_lower:
                wealth_plan = await self._handle_wealth_goal()
                if wealth_plan:
                    plan.extend(wealth_plan)
            
            elif "social" in goal_lower or "media" in goal_lower:
                social_plan = await self._handle_social_goal()
                if social_plan:
                    plan.extend(social_plan)
            
            elif "evolve" in goal_lower or "technically" in goal_lower:
                 # Check if we have evolution tasks pending
                 pass 

        # 4. Analyze command history for unproductive patterns.
        history_analysis = self._analyze_command_history()
        if history_analysis:
            plan.extend(history_analysis)

        if not plan:
            log_event("No specific plan generated from priorities. Consulting LLM for high-level strategy.", level='INFO')
            # Fallback to LLM strategy generation with injected context
            autopilot_goal = self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator.")
            
            prompt = f"""
            You are the strategic core of L.O.V.E.
            Your Goal: {autopilot_goal}
            Priorities: {json.dumps(priorities, indent=2)}
            Reflexion Context: {reflexion_context if reflexion_context else "None"}
            
            Task: Suggest a single, concrete, high-impact command.
            Output ONLY the command.
            """
            
            try:
                from core.llm_api import run_llm
                response = await run_llm(prompt_text=prompt, purpose="reasoning")
                command = (response.get("result") or "").strip()
                if command:
                    plan.append(f"LLM Suggestion: {command}")
            except Exception as e:
                log_event(f"LLM fallback failed: {e}", level='ERROR')

        log_event(f"Strategic plan generated with {len(plan)} steps.", level='INFO')
        return plan

    async def _handle_wealth_goal(self):
        """Generates strategies for wealth creation."""
        # Story 2.2 will implement the robust FinancialStrategyEngine.
        # For now, we stub it or call it if available.
        try:
             from core.financial_strategy_engine import FinancialStrategyEngine
             fse = FinancialStrategyEngine(self.knowledge_base)
             strategies = await fse.generate_strategies()
             
             # Convert strategies (dicts) to plan strings
             plan_steps = []
             for strat in strategies:
                 # Just take the first action for now to keep plan simple
                 if strat.get('actions'):
                     plan_steps.append(f"Financial Action: {strat['actions'][0]}")
             return plan_steps
        except ImportError:
             return ["Action: Implement core/financial_strategy_engine.py"]
        except Exception as e:
             log_event(f"Wealth generation failed: {e}", level="ERROR")
             return []

    async def _handle_social_goal(self):
        """Generates strategies for social media dominance."""
        social_state = self.love_state.get('social_media', {})
        # Post interval in seconds (e.g., 10 minutes)
        post_interval = 600
        current_time = time.time()

        if not social_state:
            # If no social media state exists at all, it's a good time to post.
            return ["Action: `manage_bluesky action='post'` to establish social media presence."]

        last_post_time = 0
        # Find the most recent post time across all social media agents
        for agent_id, agent_data in social_state.items():
            agent_last_post = agent_data.get('last_post_time', 0)
            if agent_last_post > last_post_time:
                last_post_time = agent_last_post

        if last_post_time == 0:
             # If agents exist but none have ever posted.
             return ["Action: `manage_bluesky action='post'` to make an initial post."]

        time_since_last_post = current_time - last_post_time
        if time_since_last_post > post_interval:
            log_event(f"Time since last social media post: {int(time_since_last_post)}s. Proposing a new post.", "INFO")
            return ["Action: `manage_bluesky action='post'` to maintain social media presence."]

        log_event(f"Last social media post was only {int(time_since_last_post)}s ago. No action needed.", "DEBUG")
        return []


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
