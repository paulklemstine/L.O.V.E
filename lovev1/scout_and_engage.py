import json
from core.talent_utils.aggregator import process_domain_data

def scout_and_engage():
    """
    A demonstration of the process_domain_data function for scouting and engaging
    potential talent based on specified keywords and a domain.
    """
    # 1. Define a list of keywords relevant to scouting and engagement.
    keywords = ["fashion", "sustainability", "creative", "art"]

    # 2. Specify the target domain for the search.
    domain = "tiktok"

    print(f"Scouting on domain: {domain} with keywords: {keywords}")

    # 3. Call the process_domain_data function.
    scouted_profiles = process_domain_data(keywords, domain)

    # 4. Analyze the structured output.
    print("\n--- Scouted Profiles ---")
    if scouted_profiles:
        print(json.dumps(scouted_profiles, indent=4))
    else:
        print("No profiles found or an error occurred.")

    # Example with a custom filter
    def filter_by_followers(profiles, min_followers=1000):
        """A simple filter to demonstrate custom filtering logic."""
        filtered_list = []
        for profile in profiles:
            if profile.get('followers_count', 0) > min_followers:
                filtered_list.append(profile)
        return filtered_list

    print(f"\nScouting again on domain: {domain} with keywords: {keywords} and a follower filter (>1000)")
    scouted_and_filtered_profiles = process_domain_data(keywords, domain, parsing_and_filtering_logic=filter_by_followers)

    print("\n--- Scouted and Filtered Profiles ---")
    if scouted_and_filtered_profiles:
        print(json.dumps(scouted_and_filtered_profiles, indent=4))
    else:
        print("No profiles found that match the filter criteria.")


if __name__ == "__main__":
    scout_and_engage()
