import os
import hashlib
import json
import requests
from box import Box
import httpx
from parsel import Selector
from atproto import Client, models
from atproto_client.models.app.bsky.feed import get_author_feed, search_posts
from core.logging import log_event
import datetime
import secrets
from urllib.parse import urlencode, quote

class EthicalFilterBundle:
    def __init__(self, min_followers=0, require_verified=False, min_sentiment=0.0):
        self.min_followers = min_followers
        self.require_verified = require_verified
        self.min_sentiment = min_sentiment

    def filter_profiles(self, profiles):
        """Filters a list of profiles based on the defined criteria."""
        filtered_profiles = []
        for profile in profiles:
            if self._passes_filters(profile):
                filtered_profiles.append(profile)
        return filtered_profiles

    def _passes_filters(self, profile):
        """Checks if a single profile passes all defined filters."""
        if profile.get('followers_count', 0) < self.min_followers:
            return False
        if self.require_verified and not profile.get('verified', False):
            return False
        # Placeholder for sentiment analysis
        # once the Analyzer is implemented, we can calculate the sentiment of the profile's bio and posts
        # and compare it to self.min_sentiment
        return True

class PublicProfileAggregator:
    """
    Scrapes and collects publicly available profile data from specified platforms.
    """

    def __init__(self, keywords, platform_names, ethical_filters):
        self.keywords = keywords
        self.platform_names = platform_names
        self.ethical_filters = ethical_filters
        self.client = self._get_bluesky_client()
        self.http_client = httpx.AsyncClient(
            http2=True,
            headers={
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
            }
        )

    def _get_bluesky_client(self):
        """Initializes and returns a Bluesky client if credentials are available."""
        bluesky_user = os.environ.get("BLUESKY_USER")
        bluesky_password = os.environ.get("BLUESKY_PASSWORD")

        if not (bluesky_user and bluesky_password):
            log_event("Bluesky credentials (BLUESKY_USER, BLUESKY_PASSWORD) not found in environment variables.", level='WARNING')
            return None

        try:
            client = Client()
            client.login(bluesky_user, bluesky_password)
            return client
        except Exception as e:
            log_event(f"Error connecting to Bluesky: {e}", level='ERROR')
            return None

    def _anonymize_id(self, user_id):
        """Hashes the user's ID to create a non-reversible, anonymized ID."""
        return hashlib.sha256(user_id.encode('utf-8')).hexdigest()

    async def _get_bluesky_posts(self, user_did, limit=25):
        """Fetches recent posts for a given Bluesky user DID."""
        try:
            params = get_author_feed.Params(actor=user_did, limit=limit)
            author_feed = await self.client.app.bsky.feed.get_author_feed(params)
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
            log_event(f"Error fetching Bluesky posts for DID {user_did}: {e}", level='ERROR')
            return []

    async def _search_bluesky(self, keyword):
        """Searches for profiles on Bluesky and aggregates their posts."""
        profiles = {}
        try:
            params = search_posts.Params(q=keyword, limit=100)
            search_posts_response = await self.client.app.bsky.feed.search_posts(params)
            for post_view in search_posts_response.posts:
                author = post_view.author
                post_record = post_view.record

                # If we haven't seen this author before, create a profile entry
                if author.did not in profiles:
                    profiles[author.did] = {
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
                        'posts': []
                    }

                # Add the current post to the author's post list
                if isinstance(post_record, models.AppBskyFeedPost):
                    profiles[author.did]['posts'].append({
                        'text': post_record.text,
                        'created_at': post_record.created_at,
                        'uri': post_view.uri
                    })

        except Exception as e:
            log_event(f"Error searching Bluesky for keyword '{keyword}': {e}", level='ERROR')

        return list(profiles.values())

    async def _get_instagram_profile(self, username):
        """Scrapes a single public profile from Instagram by username."""
        headers = {
            "x-ig-app-id": "936619743392459",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        try:
            response = await self.http_client.get(f'https://i.instagram.com/api/v1/users/web_profile_info/?username={username}', headers=headers)
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
            if e.response is not None:
                log_event(f"Error scraping Instagram user '{username}': {e.response.status_code} {e.response.text}", level='ERROR')
            else:
                log_event(f"Error scraping Instagram user '{username}': {e}", level='ERROR')
            return None
        except Exception as e:
            log_event(f"An unexpected error occurred while scraping Instagram user '{username}': {e}", level='ERROR')
            return None

    async def _search_instagram(self, keyword):
        """Searches for users on Instagram by keyword and scrapes their profiles."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        profiles = []
        try:
            # Use the web search endpoint
            response = await self.http_client.get(f'https://www.instagram.com/web/search/topsearch/?query={keyword}', headers=headers)
            response.raise_for_status()
            search_results = response.json()

            # Limit to the top 10 users to avoid being rate-limited
            for item in search_results.get('users', [])[:10]:
                user = item.get('user')
                if user and user.get('username'):
                    # Fetch the full profile details for each user found
                    profile_data = await self._get_instagram_profile(user['username'])
                    if profile_data:
                        profiles.append(profile_data)
            return profiles

        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                log_event(f"Error searching Instagram for keyword '{keyword}': {e.response.status_code} {e.response.text}", level='ERROR')
            else:
                log_event(f"Error searching Instagram for keyword '{keyword}': {e}", level='ERROR')
            return []
        except Exception as e:
            log_event(f"An unexpected error occurred while searching Instagram for '{keyword}': {e}", level='ERROR')
            return []


    async def _search_tiktok_by_keyword(self, keyword, max_profiles=20):
        """Searches for users on TikTok by keyword using its internal API."""
        profiles = []
        try:
            # TikTok's search API requires a session with valid cookies.
            # We obtain this by making an initial request to the search page.
            session_response = await self.http_client.get(f"https://www.tiktok.com/search?q={quote(keyword)}")
            session_response.raise_for_status()

            search_id = self._generate_tiktok_search_id()
            api_url = f"https://www.tiktok.com/api/search/general/full/?{urlencode({'keyword': keyword, 'offset': 0, 'search_id': search_id})}"

            headers = {"Referer": f"https://www.tiktok.com/search?q={quote(keyword)}"}
            api_response = await self.http_client.get(api_url, cookies=session_response.cookies, headers=headers)
            api_response.raise_for_status()

            try:
                search_results = api_response.json()
            except json.JSONDecodeError:
                log_event(f"Failed to decode JSON from TikTok search API for keyword '{keyword}'. Response text: {api_response.text}", level='ERROR')
                return []

            for item in search_results.get('data', []):
                if item.get('type') == 1:  # Type 1 corresponds to user profiles
                    user_info = item.get('item', {}).get('author', {})
                    if user_info:
                        profile = {
                            'anonymized_id': self._anonymize_id(user_info.get('id')),
                            'platform': 'tiktok',
                            'handle': user_info.get('uniqueId'),
                            'display_name': user_info.get('nickname'),
                            'bio': user_info.get('signature'),
                            'avatar_url': user_info.get('avatarLarger'),
                            'followers_count': user_info.get('followerCount'),
                            'follows_count': user_info.get('followingCount'),
                            'posts_count': user_info.get('videoCount'),
                            'source_id': user_info.get('id')
                        }
                        profiles.append(profile)
                if len(profiles) >= max_profiles:
                    break
            return profiles
        except httpx.HTTPStatusError as e:
            log_event(f"Error searching TikTok for keyword '{keyword}': {e}", level='ERROR')
            return []
        except Exception as e:
            log_event(f"An unexpected error occurred while searching TikTok for '{keyword}': {e}", level='ERROR')
            return []

    def _generate_tiktok_search_id(self):
        """Generates a random search ID in the format TikTok's API expects."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        random_hex_length = (32 - len(timestamp)) // 2
        random_hex = secrets.token_hex(random_hex_length).upper()
        return timestamp + random_hex

    async def search_and_collect(self):
        """
        Asynchronously searches for posts and profiles on specified platforms based on keywords and collects data.
        """
        all_profiles = []
        for platform in self.platform_names:
            for keyword in self.keywords:
                if platform == "bluesky":
                    if self.client:
                        all_profiles.extend(await self._search_bluesky(keyword))
                elif platform == "instagram":
                    all_profiles.extend(await self._search_instagram(keyword))
                elif platform == "tiktok":
                    all_profiles.extend(await self._search_tiktok_by_keyword(keyword))
        return self.ethical_filters.filter_profiles(all_profiles)