from typing import Dict
from core.agents.specialist_agent import SpecialistAgent
from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle
from core.talent_utils.analyzer import TraitAnalyzer, AestheticScorer, ProfessionalismRater
from core.talent_utils.manager import TalentManager
from core.talent_utils.matcher import OpportunityMatcher

class TalentAgent(SpecialistAgent):
    """
    A specialist agent that orchestrates the entire talent acquisition and
    management lifecycle.
    """

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Runs a full scout -> analyze -> engage cycle based on task details.

        Args:
            task_details: A dictionary containing all necessary parameters:
                          - opportunity_id (str)
                          - contact_templates (dict)
                          - keywords (list)
                          - platforms (list)
                          - filter_config (dict)
                          - min_score (float)
                          - engagement_type (str)

        Returns:
            A dictionary with the status and result of the cycle.
        """
        print(f"--- Starting Talent Scout for Opportunity: {task_details.get('opportunity_id')} ---")

        # Extract parameters from task_details
        opportunity_id = task_details.get("opportunity_id")
        contact_templates = task_details.get("contact_templates")
        keywords = task_details.get("keywords")
        platforms = task_details.get("platforms")
        filter_config = task_details.get("filter_config")
        min_score = task_details.get("min_score")
        engagement_type = task_details.get("engagement_type")

        if not all([opportunity_id, contact_templates, keywords, platforms, filter_config, min_score, engagement_type]):
            return {"status": "failure", "result": "Missing required parameters in task_details for TalentAgent."}

        # Initialize utility classes within the task execution
        analyzer = TraitAnalyzer(scorers={
            "aesthetic": AestheticScorer(),
            "professionalism": ProfessionalismRater()
        })
        manager = TalentManager()
        matcher = OpportunityMatcher()
        matcher.add_opportunity({"id": opportunity_id, "description": "A cool project"})

        # 1. Scout
        print(f"Scouting for talent with keywords: {keywords} on {platforms}")
        ethical_filters = EthicalFilterBundle(**filter_config)
        aggregator = PublicProfileAggregator(
            keywords=keywords,
            platform_names=platforms,
            ethical_filters=ethical_filters
        )
        profiles = aggregator.search_and_collect()
        print(f"Found {len(profiles)} initial profiles.")

        # 2. Analyze and Select
        print("Analyzing profiles...")
        selected_candidates = []
        for profile in profiles:
            posts = profile.get('posts', [])
            scores = await analyzer.analyze(profile, posts)
            profile['scores'] = scores
            average_score = sum(scores.values()) / len(scores) if scores else 0
            if average_score >= min_score:
                selected_candidates.append(profile)
                print(f"  - Selected {profile.get('handle')} with score {average_score:.2f}")

        print(f"Selected {len(selected_candidates)} candidates after analysis.")
        if not selected_candidates:
            result_message = "No candidates met the criteria. Cycle ends."
            print(result_message)
            return {"status": "success", "result": result_message}

        # 3. Engage
        print("Engaging with selected candidates...")
        for candidate in selected_candidates:
            message = manager.record_outreach(
                profile_id=candidate.get('anonymized_id'),
                message_type=engagement_type,
                dynamic_slots={"handle": candidate.get('handle'), "opportunity_id": opportunity_id}
            )
            if message:
                print(f"  - Generated outreach message for {candidate.get('handle')}.")
                print(f"    Message: {message}")

        result_message = f"Talent cycle complete. Engaged with {len(selected_candidates)} candidates."
        print(result_message)
        return {"status": "success", "result": result_message}
