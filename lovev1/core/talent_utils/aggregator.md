# PublicProfileAggregator Documentation

This document outlines the functionality and evolution of the `PublicProfileAggregator` module.

## Task History

### 2025-10-27: Instagram and TikTok Integration

*   **Task Request:** Enhance the `talent_scout` functionality by integrating new modules for aggregating public profiles from Instagram and TikTok, ensuring the system can handle the unique data structures and content formats of each platform while respecting ethical data collection standards.
*   **Pull Request:** [Link to PR will be added upon submission]
*   **Commit Hash:** [Link to commit will be added upon submission]

#### Summary of Changes

The `PublicProfileAggregator` was enhanced to support scraping public profile data from Instagram and TikTok, in addition to the existing Bluesky functionality.

*   **Instagram:** A new `_search_instagram` method was added. It uses the `requests` library to query Instagram's internal, non-public API endpoint (`/api/v1/users/web_profile_info/`) to fetch profile data by username.
*   **TikTok:** A new `_search_tiktok` method was added. It uses the `httpx` and `parsel` libraries to scrape the user's profile page. It extracts a hidden JSON object embedded in a `<script>` tag with the ID `__UNIVERSAL_DATA_FOR_REHYDRATION__` to get the profile information.
*   **Generalization:** The `search_and_collect` method was updated to iterate through the specified platforms and call the appropriate search function. It now uses the `keyword` as a username for Instagram and TikTok searches.
*   **Dependencies:** The `requirements.txt` file was updated to include `python-box`, `httpx`, and `parsel`.
*   **Data Normalization:** Data extracted from all three platforms is normalized into a consistent dictionary structure before being returned.
*   **Anonymization:** A generic `_anonymize_id` function now hashes the platform-specific user ID to create a consistent, anonymized identifier.