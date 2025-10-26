import os
from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle
from core.talent_utils.analyzer import TraitAnalyzer, AestheticScorer, ProfessionalismRater
from core.talent_utils.manager import ContactManager
from core.talent_utils.matcher import OpportunityMatcher, encrypt_params

class TalentAgent:
    """
    Orchestrates the talent acquisition and management system by integrating
    the various modules from talent_utils.
    """

    def __init__(self, opportunity_id, contact_templates):
        """
        Initializes the TalentAgent with a specific opportunity and message templates.

        Args:
            opportunity_id (str): The unique identifier for the opportunity.
            contact_templates (dict): A dictionary of message templates for outreach.
        """
        self.opportunity_id = opportunity_id
        self.contact_templates = contact_templates

        # Initialize the utility classes
        self.analyzer = TraitAnalyzer(scorers={
            "aesthetic": AestheticScorer(),
            "professionalism": ProfessionalismRater()
        })
        self.manager = ContactManager(templates=contact_templates, constraints={
            "max_attempts": 3,
            "min_response_window": "7 days"
        })
        self.matcher = OpportunityMatcher()
        self.matcher.add_opportunity({"id": self.opportunity_id, "description": "A cool project"})

    def scout(self, keywords, platforms, ethical_filter_config):
        """
        Finds potential candidates based on keywords and platforms.

        Args:
            keywords (list): A list of keywords to search for.
            platforms (list): A list of platforms to search on (e.g., ['bluesky']).
            ethical_filter_config (dict): Configuration for the ethical filters.

        Returns:
            A list of profile dictionaries.
        """
        print(f"Scouting for talent with keywords: {keywords} on {platforms}")
        ethical_filters = EthicalFilterBundle(**ethical_filter_config)
        aggregator = PublicProfileAggregator(
            keywords=keywords,
            platform_names=platforms,
            ethical_filters=ethical_filters
        )
        profiles = aggregator.search_and_collect()
        print(f"Found {len(profiles)} initial profiles.")
        return profiles

    def analyze_and_select(self, profiles, min_score_threshold):
        """
        Analyzes a list of profiles and selects the most promising candidates.

        Args:
            profiles (list): A list of profile dictionaries from the scout phase.
            min_score_threshold (float): The minimum average score to be selected.

        Returns:
            A list of selected profiles with their scores.
        """
        print("Analyzing profiles...")
        selected_candidates = []
        for profile in profiles:
            # In a real scenario, we would fetch the user's posts.
            # For now, we'll pass an empty list.
            posts = []
            scores = self.analyzer.analyze(profile, posts)
            profile['scores'] = scores

            # Simple selection logic: average score must meet the threshold
            average_score = sum(scores.values()) / len(scores) if scores else 0
            if average_score >= min_score_threshold:
                selected_candidates.append(profile)
                print(f"  - Selected {profile.get('handle')} with score {average_score:.2f}")

        print(f"Selected {len(selected_candidates)} candidates after analysis.")
        return selected_candidates

    def engage(self, selected_candidates, engagement_type):
        """
        Initiates contact with selected candidates.

        Args:
            selected_candidates (list): A list of profiles that passed analysis.
            engagement_type (str): The type of message to send (e.g., 'initial_outreach').
        """
        print("Engaging with selected candidates...")
        for candidate in selected_candidates:
            anonymized_id = candidate.get('anonymized_id')
            handle = candidate.get('handle')

            # Use the ContactManager to safely attempt outreach
            message = self.manager.record_outreach(
                profile_id=anonymized_id,
                message_type=engagement_type,
                dynamic_slots={"handle": handle, "opportunity_id": self.opportunity_id}
            )

            if message:
                print(f"  - Generated outreach message for {handle}.")
                # Here, you would plug into an actual delivery mechanism.
                # For now, we just print the message.
                print("    Message:", message)

    def full_cycle(self, keywords, platforms, filter_config, min_score, engagement_type):
        """
        Runs a full scout -> analyze -> engage cycle.
        """
        # 1. Scout
        profiles = self.scout(keywords, platforms, filter_config)

        # 2. Analyze and Select
        selected = self.analyze_and_select(profiles, min_score)
        if not selected:
            print("No candidates met the criteria. Cycle ends.")
            return

        # 3. Engage
        self.engage(selected, engagement_type)

        print("\nTalent cycle complete.")
        # Optional: Log bias warnings if any were detected
        bias_warnings = self.analyzer.detect_bias()
        if bias_warnings:
            print("\n--- BIAS WARNINGS ---")
            for warning in bias_warnings:
                print(warning)

# This function would be registered with the ToolRegistry in the main application
def talent_scout(keywords, platforms, opportunity_id="default_opportunity_123"):
    """
    A high-level function to discover, analyze, and engage new talent for a given opportunity.

    Args:
        keywords (list): A list of strings containing keywords to search for.
        platforms (list): A list of platforms to search on (e.g., ['bluesky']).
        opportunity_id (str): The unique identifier for the opportunity.
    """
    print(f"--- Starting Talent Scout for Opportunity: {opportunity_id} ---")

    # Define message templates for this run
    templates = {
        "initial_outreach": "Hi [handle], we were impressed by your profile and have an opportunity ([opportunity_id]) we think you'd be a great fit for.",
        "follow_up": "Hi [handle], just following up on our previous message about opportunity [opportunity_id]."
    }

    # Define ethical filters for the search
    filter_config = {
        "min_sentiment": 0.2,
        "required_tags": ["art", "design"],
        "privacy_level": "public"
    }

    # Instantiate and run the agent
    agent = TalentAgent(opportunity_id=opportunity_id, contact_templates=templates)
    agent.full_cycle(
        keywords=keywords,
        platforms=platforms,
        filter_config=filter_config,
        min_score=0.6,
        engagement_type="initial_outreach"
    )

    print(f"--- Talent Scout cycle finished for Opportunity: {opportunity_id} ---")
    return "Talent scout cycle finished. Check logs for details."