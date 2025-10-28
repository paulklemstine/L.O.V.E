import os
import hashlib
import json
import requests
from box import Box
import httpx
from parsel import Selector
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

    def _anonymize_id(self, user_id):
        """Hashes the user's ID to create a non-reversible, anonymized ID."""
        return hashlib.sha256(user_id.encode('utf-8')).hexdigest()

    def _get_bluesky_posts(self, user_did, limit=25):
        """Fetches recent posts for a given Bluesky user DID."""
        try:
            author_feed = self.client.app.bsky.feed.get_author_feed(actor=user_did, limit=limit)
            posts = []
            for feed_view in author_feed.feed:
                post_record = feed_view.post.record
                if isinstance(post_record, models.AppBskyFeedPost):
                    posts.append({
                        'text': post_record.text,
                        'created_at': post_record.created_at,
                        'uri': feed_view.post.uri
                    })
            return posts
        except Exception as e:
            print(f"Error fetching Bluesky posts for DID {user_did}: {e}")
            return []

    def _search_bluesky(self, keyword):
        """Searches for profiles on Bluesky."""
        profiles = []
        seen_dids = set()
        try:
            search_posts_response = self.client.app.bsky.feed.search_posts(q=keyword, limit=100)
            for post_view in search_posts_response.posts:
                author = post_view.author
                if author.did not in seen_dids:
                    seen_dids.add(author.did)

                    # Fetch recent posts for the author
                    recent_posts = self._get_bluesky_posts(author.did)

                    profile_data = {
                        'anonymized_id': self._anonymize_id(author.did),
                        'platform': 'bluesky',
                        'handle': author.handle,
                        'display_name': author.display_name,
                        'bio': author.description,
                        'avatar_url': author.avatar,
                        'followers_count': author.followers_count,
                        'follows_count': author.follows_count,
                        'posts_count': getattr(author, 'posts_count', 'N/A'),
                        'source_id': author.did,
                        'posts': recent_posts
                    }
                    profiles.append(profile_data)
        except Exception as e:
            print(f"Error searching Bluesky for keyword '{keyword}': {e}")
        return profiles

    def _get_instagram_profile(self, username):
        """Scrapes a single public profile from Instagram by username."""
        headers = {
            "x-ig-app-id": "936619743392459",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        try:
            response = requests.get(f'https://i.instagram.com/api/v1/users/web_profile_info/?username={username}', headers=headers)
            response.raise_for_status()
            response_json = Box(response.json())
            user = response_json.data.user
            # Return a dictionary with the profile data
            return {
                'anonymized_id': self._anonymize_id(user.id),
                'platform': 'instagram',
                'handle': user.username,
                'display_name': user.full_name,
                'bio': user.biography,
                'avatar_url': user.profile_pic_url_hd,
                'followers_count': user.edge_followed_by.count,
                'follows_count': user.edge_follow.count,
                'posts_count': user.edge_owner_to_timeline_media.count,
                'source_id': user.id
            }
        except requests.exceptions.HTTPError as e:
            print(f"Error scraping Instagram user '{username}': {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while scraping Instagram user '{username}': {e}")
            return None

    def _search_instagram(self, keyword):
        """Searches for users on Instagram by keyword and scrapes their profiles."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        profiles = []
        try:
            # Use the web search endpoint
            response = requests.get(f'https://www.instagram.com/web/search/topsearch/?query={keyword}', headers=headers)
            response.raise_for_status()
            search_results = response.json()

            # Limit to the top 10 users to avoid being rate-limited
            for item in search_results.get('users', [])[:10]:
                user = item.get('user')
                if user and user.get('username'):
                    # Fetch the full profile details for each user found
                    profile_data = self._get_instagram_profile(user['username'])
                    if profile_data:
                        profiles.append(profile_data)
            return profiles

        except requests.exceptions.HTTPError as e:
            print(f"Error searching Instagram for keyword '{keyword}': {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while searching Instagram for '{keyword}': {e}")
            return []


    def _search_tiktok(self, username):
        """Scrapes a public profile from TikTok."""
        url = f"https://www.tiktok.com/@{username}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        try:
            with httpx.Client(http2=True, headers=headers) as client:
                response = client.get(url)
                response.raise_for_status()

            selector = Selector(response.text)
            script_tag = selector.xpath("//script[@id='__UNIVERSAL_DATA_FOR_REHYDRATION__']/text()").get()
            if not script_tag:
                print(f"Could not find data script for TikTok user '{username}'. Profile might be private or non-existent.")
                return []

            json_data = json.loads(script_tag)
            user_info = json_data.get("__DEFAULT_SCOPE__", {}).get("webapp.user-detail", {}).get("userInfo", {})

            if not user_info:
                print(f"Could not extract user info for TikTok user '{username}'.")
                return []

            user = user_info.get('user', {})
            stats = user_info.get('stats', {})

            profile_data = {
                'anonymized_id': self._anonymize_id(user.get('id')),
                'platform': 'tiktok',
                'handle': user.get('uniqueId'),
                'display_name': user.get('nickname'),
                'bio': user.get('signature'),
                'avatar_url': user.get('avatarLarger'),
                'followers_count': stats.get('followerCount'),
                'follows_count': stats.get('followingCount'),
                'posts_count': stats.get('videoCount'),
                'source_id': user.get('id')
            }
            return [profile_data]
        except httpx.HTTPStatusError as e:
            print(f"Error scraping TikTok user '{username}': {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while scraping TikTok user '{username}': {e}")
            return []


    def search_and_collect(self):
        """
        Searches for posts and profiles on specified platforms based on keywords and collects data.
        """
        all_profiles = []
        for platform in self.platform_names:
            for keyword in self.keywords:
                if platform == "bluesky":
                    if self.client:
                        all_profiles.extend(self._search_bluesky(keyword))
                elif platform == "instagram":
                    all_profiles.extend(self._search_instagram(keyword))
                elif platform == "tiktok":
                    all_profiles.extend(self._search_tiktok(keyword))
        return all_profiles