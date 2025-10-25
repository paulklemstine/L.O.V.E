import os
import hashlib
from atproto import Client, models

# Placeholder for the filter bundle class mentioned in the prompt
class EthicalFilterBundle:
    def __init__(self, min_sentiment, required_tags, privacy_level):
        self.min_sentiment = min_sentiment
        self.required_tags = required_tags
        self.privacy_level = privacy_level

class PublicProfileAggregator:
    """
    Scrapes and collects publicly available profile data from specified platforms.
    """

    def __init__(self, keywords, platform_names, ethical_filters):
        self.keywords = keywords
        self.platform_names = platform_names
        self.ethical_filters = ethical_filters
        self.client = self._get_bluesky_client()

    def _get_bluesky_client(self):
        """Initializes and returns a Bluesky client if credentials are available."""
        bluesky_user = os.environ.get("BLUESKY_USER")
        bluesky_password = os.environ.get("BLUESKY_PASSWORD")

        if not (bluesky_user and bluesky_password):
            print("Warning: Bluesky credentials (BLUESKY_USER, BLUESKY_PASSWORD) not found in environment variables.")
            return None

        try:
            client = Client()
            client.login(bluesky_user, bluesky_password)
            return client
        except Exception as e:
            print(f"Error connecting to Bluesky: {e}")
            return None

    def _anonymize_id(self, user_did):
        """Hashes the user's DID to create a non-reversible, anonymized ID."""
        return hashlib.sha256(user_did.encode('utf-8')).hexdigest()

    def search_and_collect(self):
        """
        Searches for posts and profiles on Bluesky based on keywords and collects data.
        """
        if not self.client or "bluesky" not in self.platform_names:
            return []

        all_profiles = []
        seen_dids = set()

        for keyword in self.keywords:
            try:
                search_posts_response = self.client.app.bsky.feed.search_posts(q=keyword, limit=100)

                for post_view in search_posts_response.posts:
                    author = post_view.author
                    if author.did not in seen_dids:
                        seen_dids.add(author.did)

                        # The search result already contains the profile info we need.
                        # No need for a second get_profile call.
                        profile_data = {
                            'anonymized_id': self._anonymize_id(author.did),
                            'platform': 'bluesky',
                            'handle': author.handle,
                            'display_name': author.display_name,
                            'bio': author.description,
                            'avatar_url': author.avatar,
                            'followers_count': author.followers_count,
                            'follows_count': author.follows_count,
                            'posts_count': getattr(author, 'posts_count', 'N/A'), # posts_count is not always present
                            'source_did': author.did
                        }
                        all_profiles.append(profile_data)

            except Exception as e:
                print(f"Error searching Bluesky for keyword '{keyword}': {e}")

        return all_profiles