import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import json
import sys

# Mock logging before any other imports from our application
sys.modules['core.logging'] = MagicMock()

# Import the concrete model class for instantiation
from atproto_client.models.app.bsky.feed.post import Record as AppBskyFeedPostRecord
# Import the top-level models for the isinstance check to work with our mock
from atproto import models

# Now we can import the modules we want to test
from core.talent_utils.opportunity_scraper import OpportunityScraper
from core.talent_utils.opportunity_matcher import OpportunityMatcher

class TestOpportunitySystem(unittest.IsolatedAsyncioTestCase):

    @patch.dict(os.environ, {"BLUESKY_USER": "testuser", "BLUESKY_PASSWORD": "testpassword"})
    @patch('core.talent_utils.opportunity_scraper.Client') # Patch where it's used
    def test_opportunity_scraper_success(self, MockBlueskyClient):
        """
        Tests if the OpportunityScraper correctly processes a successful API response.
        """
        # --- Arrange ---
        mock_client_instance = MockBlueskyClient.return_value
        mock_client_instance.login.return_value = True

        mock_post_view = MagicMock()
        mock_post_view.author.handle = 'test.bsky.social'
        mock_post_view.author.did = 'did:plc:test'
        mock_post_view.author.display_name = 'Test Author'
        mock_post_view.author.avatar = 'http://example.com/avatar.jpg'
        mock_post_view.uri = 'at://did:plc:test/app.bsky.feed.post/12345'

        # FIX: Instantiate the concrete Record class, which will pass the isinstance check
        mock_record = AppBskyFeedPostRecord(text='This is a test opportunity for a photographer.', created_at='2024-01-01T00:00:00Z')
        mock_post_view.record = mock_record

        mock_api_response = MagicMock()
        mock_api_response.posts = [mock_post_view]
        mock_client_instance.app.bsky.feed.search_posts.return_value = mock_api_response

        # --- Act ---
        scraper = OpportunityScraper(keywords=['photographer'])
        opportunities = scraper.search_for_opportunities()

        # --- Assert ---
        self.assertEqual(len(opportunities), 1)
        opportunity = opportunities[0]
        self.assertEqual(opportunity['text'], 'This is a test opportunity for a photographer.')
        self.assertEqual(opportunity['author_handle'], 'test.bsky.social')
        mock_client_instance.app.bsky.feed.search_posts.assert_called_once()

    @patch('core.talent_utils.opportunity_matcher.run_llm', new_callable=AsyncMock)
    async def test_opportunity_matcher_finds_match(self, mock_run_llm):
        """
        Tests if the OpportunityMatcher correctly identifies and processes a match.
        """
        # --- Arrange ---
        mock_llm_response = {
            "is_match": True,
            "match_score": 95,
            "reasoning": "Directly relevant.",
            "opportunity_type": "Paid Gig"
        }
        mock_run_llm.return_value = (json.dumps(mock_llm_response), None)

        opportunities = [{'text': 'Seeking a photographer.', 'author_did': 'did:a'}]
        talent_profiles = [{'bio': 'I am a photographer.', 'source_id': 'did:b', 'handle': 'photo.bsky.social'}]

        # --- Act ---
        matcher = OpportunityMatcher(talent_profiles=talent_profiles)
        matches = await matcher.find_matches(opportunities)

        # --- Assert ---
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['match_evaluation']['match_score'], 95)
        mock_run_llm.assert_called_once()

    @patch('core.talent_utils.opportunity_matcher.run_llm', new_callable=AsyncMock)
    async def test_opportunity_matcher_skips_no_match(self, mock_run_llm):
        """
        Tests if the OpportunityMatcher correctly skips a non-match.
        """
        # --- Arrange ---
        mock_llm_response = {"is_match": False}
        mock_run_llm.return_value = (json.dumps(mock_llm_response), None)

        opportunities = [{'text': 'We need a developer.', 'author_did': 'did:a'}]
        talent_profiles = [{'bio': 'I am a photographer.', 'source_id': 'did:b'}]

        # --- Act ---
        matcher = OpportunityMatcher(talent_profiles=talent_profiles)
        matches = await matcher.find_matches(opportunities)

        # --- Assert ---
        self.assertEqual(len(matches), 0)
        mock_run_llm.assert_called_once()

if __name__ == '__main__':
    unittest.main()
