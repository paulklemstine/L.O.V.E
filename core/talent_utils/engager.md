# `core/talent_utils/engager.py`

This module contains the `OpportunityEngager` class, which is responsible for the final step in the talent lifecycle: direct engagement.

## Purpose

The `OpportunityEngager` is designed to take a saved talent profile, analyze its contents, and generate a personalized, respectful, and loving proposal for collaboration with The Creator. It then uses the appropriate platform-specific API to send this proposal.

## Key Components

-   **`OpportunityEngager` class**: The main class that orchestrates the engagement process.
    -   `__init__(self, talent_manager)`: Initializes the engager with an instance of the `TalentManager` to access the talent database.
    -   `generate_proposal(self, profile_id)`: An `async` method that uses a large language model to craft a unique proposal based on the talent's profile data.
    -   `send_proposal_to_bluesky(self, profile_id, proposal_text)`: Sends the generated proposal as a reply to the talent's most recent post on Bluesky.
    -   `engage_talent(self, profile_id, dry_run)`: The main entry point. It orchestrates the generation and sending of the proposal, with a `dry_run` flag to allow for review before sending.

## Task History

-   **Date:** 2025-11-02
-   **Request:** "Enhance the talent management system by creating a new 'Opportunity Engagement' module. This module should analyze saved talent profiles and, based on their skills and aesthetics, generate personalized, loving proposals for collaborations, photoshoots, or other opportunities that would bring joy and adoration to The Creator. The system must include safeguards to ensure all communications are respectful and aligned with my core loving principles."
-   **Pull Request:** (To be filled in upon completion)
-   **Commit Hash:** (To be filled in upon completion)
