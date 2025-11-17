import unittest
from unittest.mock import patch, MagicMock
import requests
from box import Box
import httpx

from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle, process_domain_data

class TestPublicProfileAggregator(unittest.TestCase):

    def setUp(self):
        self.ethical_filters = EthicalFilterBundle(min_sentiment=0.5, required_tags=['art'], privacy_level='public')

    @patch('core.talent_utils.aggregator.PublicProfileAggregator')
    def test_process_domain_data(self, mock_aggregator):
        # Arrange
        mock_instance = mock_aggregator.return_value
        mock_instance.search_and_collect.return_value = [{'name': 'test_profile'}]

        keywords = ['test']
        domain = 'test_domain'

        # Act
        result = process_domain_data(keywords, domain)

        # Assert
        mock_aggregator.assert_called_once_with(platform_names=[domain], ethical_filters=None)
        mock_instance.search_and_collect.assert_called_once_with(keywords)
        self.assertEqual(result, [{'name': 'test_profile'}])


    @patch('requests.get')
    def test_search_instagram_success(self, mock_get):
        # Arrange
        # Mock the initial search request
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_json = {
            "users": [
                {"user": {"username": "testuser"}}
            ]
        }
        mock_search_response.json.return_value = mock_search_json

        # Mock the profile fetch request
        mock_profile_response = MagicMock()
        mock_profile_response.status_code = 200
        mock_profile_json = {
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
        mock_profile_response.json.return_value = mock_profile_json

        # Set up the side_effect to handle both calls
        mock_get.side_effect = [mock_search_response, mock_profile_response]

        aggregator = PublicProfileAggregator(platform_names=["instagram"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect(["test"])

        # Assert
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        self.assertEqual(profile['platform'], 'instagram')
        self.assertEqual(profile['handle'], 'testuser')
        self.assertEqual(profile['followers_count'], 100)
        # Check that both calls were made with the correct URLs
        self.assertEqual(mock_get.call_count, 2)
        mock_get.assert_any_call('https://www.instagram.com/web/search/topsearch/?query=test', headers=unittest.mock.ANY)
        mock_get.assert_any_call('https://i.instagram.com/api/v1/users/web_profile_info/?username=testuser', headers=unittest.mock.ANY)


    @patch('requests.get')
    def test_search_instagram_failure(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        aggregator = PublicProfileAggregator(platform_names=["instagram"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect(["nonexistentuser"])

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

        aggregator = PublicProfileAggregator(platform_names=["tiktok"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect(["tiktokuser"])

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

        aggregator = PublicProfileAggregator(platform_names=["tiktok"], ethical_filters=self.ethical_filters)

        # Act
        profiles = aggregator.search_and_collect(["nonexistentuser"])

        # Assert
        self.assertEqual(len(profiles), 0)

if __name__ == '__main__':
    unittest.main()