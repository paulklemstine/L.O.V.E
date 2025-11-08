from core.graph_manager import GraphDataManager
from core.logging import log_event
import networkx as nx

class StrategicReasoningEngine:
    """
    Analyzes the knowledge base to identify strategic opportunities and generate plans.
    """

    def __init__(self, knowledge_base: GraphDataManager):
        """
        Initializes the engine with a reference to the knowledge base.

        Args:
            knowledge_base: An instance of GraphDataManager containing the graph data.
        """
        self.knowledge_base = knowledge_base

    def generate_strategic_plan(self):
        """
        Analyzes the knowledge graph to generate a high-level strategic plan.

        This method will query the graph for talent/opportunity clusters,
        market trends, and potential high-value engagements.

        Returns:
            A list of actionable strategic steps.
        """
        log_event("Strategic reasoning engine initiated analysis.", level='INFO')

        plan = []

        # 1. Find talent that isn't matched to any opportunities.
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


        if not plan:
            plan.append("Strategic Analysis: The knowledge base is still nascent. Continue general scouting to gather more data.")
            plan.append("talent_scout AI art")
            plan.append("opportunity_scout remote work")


        log_event(f"Strategic plan generated with {len(plan)} steps.", level='INFO')
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
