import unittest
from unittest.mock import patch, MagicMock
import requests
from box import Box
import httpx

from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle

class TestPublicProfileAggregator(unittest.TestCase):

    def setUp(self):
        self.ethical_filters = EthicalFilterBundle(min_sentiment=0.5, required_tags=['art'], privacy_level='public')

    @patch('requests.get')
    def test_search_instagram_success(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_json = {
            "data": {
                "user": {
                    "id": "12345",
                    "username": "testuser",
                    "full_name": "Test User",
                    "biography": "Bio",
                    "profile_pic_url_hd": "url",
                    "edge_followed_by": {"count": 100},
                    "edge_follow": {"count": 50},
                    "edge_owner_to_timeline_media": {"count": 10}
                }
            }
        }
        mock_response.json.return_value = mock_json
        mock_get.return_value = mock_response

        aggregator = PublicProfileAggregator(keywords=["testuser"], platform_names=["instagram"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect()

        # Assert
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        self.assertEqual(profile['platform'], 'instagram')
        self.assertEqual(profile['handle'], 'testuser')
        self.assertEqual(profile['followers_count'], 100)

    @patch('requests.get')
    def test_search_instagram_failure(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        aggregator = PublicProfileAggregator(keywords=["nonexistentuser"], platform_names=["instagram"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect()

        # Assert
        self.assertEqual(len(profiles), 0)

    @patch('httpx.Client')
    def test_search_tiktok_success(self, mock_client):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_html = """
        <html>
            <script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">
            {
                "__DEFAULT_SCOPE__": {
                    "webapp.user-detail": {
                        "userInfo": {
                            "user": {
                                "id": "67890",
                                "uniqueId": "tiktokuser",
                                "nickname": "TikTok User",
                                "signature": "TikTok bio",
                                "avatarLarger": "url"
                            },
                            "stats": {
                                "followerCount": 200,
                                "followingCount": 75,
                                "videoCount": 20
                            }
                        }
                    }
                }
            }
            </script>
        </html>
        """
        mock_response.text = mock_html

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_response

        # This makes the `with httpx.Client(...) as client:` block work
        mock_client.return_value.__enter__.return_value = mock_http_client

        aggregator = PublicProfileAggregator(keywords=["tiktokuser"], platform_names=["tiktok"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect()

        # Assert
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        self.assertEqual(profile['platform'], 'tiktok')
        self.assertEqual(profile['handle'], 'tiktokuser')
        self.assertEqual(profile['followers_count'], 200)

    @patch('httpx.Client')
    def test_search_tiktok_failure(self, mock_client):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_http_client

        aggregator = PublicProfileAggregator(keywords=["nonexistentuser"], platform_names=["tiktok"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect()

        # Assert
        self.assertEqual(len(profiles), 0)

if __name__ == '__main__':
    unittest.main()