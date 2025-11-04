from core.llm_api import run_llm
from core.logging import log_event
import json

class OpportunityMatcher:
    """
    Uses an LLM to intelligently match scraped opportunities with talent profiles.
    """

    def __init__(self, talent_profiles):
        """
        Initializes the matcher with a list of talent profiles.

        Args:
            talent_profiles (list): A list of talent profile dictionaries from TalentManager.
        """
        self.talent_profiles = talent_profiles

    def _construct_matching_prompt(self, opportunity, profile):
        """
        Creates a detailed prompt for the LLM to evaluate the match potential.
        """
        # Abridged profile for prompt efficiency
        profile_summary = {
            "handle": profile.get('handle'),
            "platform": profile.get('platform'),
            "bio": profile.get('bio'),
            "display_name": profile.get('display_name'),
            "analysis": profile.get('analysis', {}) # Include professional analysis if available
        }

        prompt = f"""
        **Task: Evaluate the match between a professional opportunity and a creative talent.**

        **Opportunity Details:**
        - **Platform:** {opportunity.get('platform')}
        - **Author:** {opportunity.get('author_handle')} ({opportunity.get('author_display_name')})
        - **Content:** "{opportunity.get('text')}"

        **Talent Profile Summary:**
        - **Handle:** {profile_summary.get('handle')} on {profile_summary.get('platform')}
        - **Bio:** "{profile_summary.get('bio')}"
        - **Professionalism Analysis:** {json.dumps(profile_summary.get('analysis'), indent=2)}

        **Analysis Instructions:**
        1.  **Relevance:** Is the opportunity relevant to the talent's professional field, skills, or stated interests in their bio?
        2.  **Aesthetic/Vibe:** Does the tone and content of the opportunity post align with the talent's personal brand and aesthetic as suggested by their profile?
        3.  **Professionalism:** Based on the talent's professionalism analysis, are they a suitable candidate for this type of opportunity? (e.g., brand collaborations require high professionalism).
        4.  **Actionability:** Is the opportunity post a clear, actionable offer or request for collaboration?

        **Output Format:**
        Provide your response as a JSON object with the following structure:
        {{
            "is_match": boolean,
            "match_score": integer (1-100, where 100 is a perfect match),
            "reasoning": "A concise explanation for your decision, covering relevance, aesthetic, professionalism, and actionability.",
            "opportunity_type": "Categorize the opportunity (e.g., 'Paid Gig', 'Collaboration', 'Networking', 'Uncertain')."
        }}
        """
        return prompt

    async def find_matches(self, opportunities):
        """
        Asynchronously evaluates a list of opportunities against the stored talent profiles.

        Args:
            opportunities (list): A list of opportunity dictionaries from OpportunityScraper.

        Returns:
            A list of match dictionaries, each containing the opportunity, the matched profile,
            and the LLM's evaluation.
        """
        log_event(f"Starting opportunity matching for {len(opportunities)} opportunities against {len(self.talent_profiles)} profiles.", level='INFO')
        all_matches = []

        for opportunity in opportunities:
            for profile in self.talent_profiles:
                # Basic filter: don't match a user to their own posts.
                if opportunity.get('author_did') == profile.get('source_id'):
                    continue

                prompt = self._construct_matching_prompt(opportunity, profile)

                try:
                    llm_response_str, error = await run_llm(prompt, is_source_code=False)
                    if error:
                        log_event(f"LLM call failed for matching. Error: {error}", level='ERROR')
                        continue

                    # The response from run_llm might be a JSON string.
                    # It's safer to parse it defensively.
                    try:
                        llm_evaluation = json.loads(llm_response_str)
                    except json.JSONDecodeError:
                         # Fallback if the LLM doesn't return perfect JSON
                        log_event(f"Could not decode LLM response as JSON: {llm_response_str}", level='WARNING')
                        # We can try to extract the boolean with a simpler LLM call or regex,
                        # but for now, we'll just skip.
                        continue

                    if llm_evaluation.get("is_match"):
                        log_event(f"Found a potential match! Score: {llm_evaluation.get('match_score')}. Opportunity: '{opportunity.get('text')[:50]}...' matched with Talent: {profile.get('handle')}", level='SUCCESS')
                        match_data = {
                            "opportunity": opportunity,
                            "talent_profile": profile,
                            "match_evaluation": llm_evaluation
                        }
                        all_matches.append(match_data)

                except Exception as e:
                    log_event(f"An unexpected error occurred during matching: {e}", level='CRITICAL')

        log_event(f"Completed matching process. Found {len(all_matches)} high-potential matches.", level='INFO')
        return all_matches
