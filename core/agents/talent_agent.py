import random
from typing import Dict, Any
from core.agents.specialist_agent import SpecialistAgent
from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle
from core.talent_utils.analyzer import TraitAnalyzer, AestheticScorer, ProfessionalismRater
from core.talent_utils.manager import TalentManager
from core.talent_utils.campaign import generate_outreach_campaign

class TalentAgent(SpecialistAgent):
    """
    A specialist agent that orchestrates the entire talent acquisition and
    management lifecycle.
    """

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plans and executes a talent outreach campaign.

        Args:
            task_details: A dictionary containing all necessary parameters for the campaign.

        Returns:
            A dictionary with the status and result of the campaign.
        """
        required_params = ["opportunity_id", "target_demographics", "engagement_strategy", "communication_templates", "performance_metrics"]
        if not all(param in task_details for param in required_params):
            return {"status": "failure", "result": "Missing required parameters in task_details for TalentAgent."}

        print(f"--- Starting Talent Campaign for Opportunity: {task_details.get('opportunity_id')} ---")

        campaign = await generate_outreach_campaign(
            target_demographics=task_details.get("target_demographics", {}),
            engagement_strategy=task_details.get("engagement_strategy", ""),
            communication_templates=task_details.get("communication_templates", []),
            performance_metrics=task_details.get("performance_metrics", [])
        )

        execution_plan = campaign.get("execution_plan", {})
        potential_targets = campaign.get("potential_targets", [])

        analyzer = TraitAnalyzer(scorers={"aesthetic": AestheticScorer(), "professionalism": ProfessionalismRater()})
        talent_manager = TalentManager()

        talent_manager.add_opportunity({"id": task_details.get("opportunity_id"), "description": "A cool project"})

        selected_candidates = []
        min_score = task_details.get("min_score", 0.7)
        for profile in potential_targets:
            posts = profile.get("posts", [])
            scores = await analyzer.analyze(profile, posts)
            if sum(scores.values()) / len(scores) if scores else 0 >= min_score:
                selected_candidates.append(profile)

        if not selected_candidates:
            return {"status": "success", "result": "No candidates met the criteria. Campaign ends."}

        communication_templates = execution_plan.get("templates", [])
        if not communication_templates:
            return {"status": "failure", "result": "No communication templates available for engagement."}

        for candidate in selected_candidates:
            message_template = random.choice(communication_templates)
            message = message_template.format(handle=candidate.get('handle'), opportunity_id=task_details.get("opportunity_id"))

            talent_manager.add_interaction(
                anonymized_id=candidate.get('anonymized_id'),
                interaction_type='outreach',
                message=message
            )
            print(f"  - Recorded outreach for {candidate.get('handle')}: {message}")

        return {"status": "success", "result": f"Talent campaign complete. Engaged with {len(selected_candidates)} candidates.", "campaign_plan": campaign}
