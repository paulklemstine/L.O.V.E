import unittest
import os
import hashlib
from unittest.mock import patch, Mock, MagicMock

# It's good practice to add the project root to the path for testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle
from core.talent_utils.analyzer import TraitAnalyzer, AestheticScorer, ProfessionalismRater
from core.talent_utils.manager import TalentManager

class TestTalentUtils(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Generate a valid base64-encoded key for testing.
        from cryptography.fernet import Fernet
        self.test_key = Fernet.generate_key().decode('utf-8')
        self.test_db_file = "test_talent_database.enc"
        # Mock environment variables
        self.patcher_env = patch.dict(os.environ, {
            "BLUESKY_USER": "testuser",
            "BLUESKY_PASSWORD": "testpassword",
            "TALENT_LOG_KEY": self.test_key
        })
        self.patcher_env.start()

    def tearDown(self):
        """Clean up test environment after each test."""
        self.patcher_env.stop()
        if os.path.exists(self.test_db_file):
            os.remove(self.test_db_file)

    @patch('core.talent_utils.aggregator.Client')
    def test_public_profile_aggregator(self, MockBlueskyClient):
        """Test the PublicProfileAggregator's search and anonymization."""
        # Mock the Bluesky client and its response
        mock_client_instance = Mock()
        mock_post_view = Mock()
        mock_author = Mock(
            did="did:plc:12345",
            handle="test.bsky.social",
            display_name="Test User",
            description="A test bio.",
            avatar="http://example.com/avatar.jpg",
            followers_count=100,
            follows_count=50
        )
        mock_post_view.author = mock_author
        mock_client_instance.app.bsky.feed.search_posts.return_value = Mock(posts=[mock_post_view])
        MockBlueskyClient.return_value = mock_client_instance

        # Initialize the aggregator
        filters = EthicalFilterBundle(min_sentiment=0.7, required_tags={"art"}, privacy_level="public_only")
        aggregator = PublicProfileAggregator(keywords=["art"], platform_names=["bluesky"], ethical_filters=filters)

        # Run the search
        profiles = aggregator.search_and_collect()

        # Assertions
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        self.assertEqual(profile['handle'], "test.bsky.social")
        self.assertEqual(profile['display_name'], "Test User")

        # Verify anonymization
        expected_id = hashlib.sha256("did:plc:12345".encode('utf-8')).hexdigest()
        self.assertEqual(profile['anonymized_id'], expected_id)

        # Verify that the login was called
        mock_client_instance.login.assert_called_with("testuser", "testpassword")

    @patch('core.talent_utils.analyzer.run_llm', new_callable=unittest.mock.AsyncMock)
    async def test_trait_analyzer(self, mock_run_llm):
        """Test the TraitAnalyzer with its scoring plugins."""
        # Mock the LLM response for the ProfessionalismRater
        mock_run_llm.return_value = {"result": "8"}

        # Setup scorers
        scorers = {
            "aesthetics": AestheticScorer(),
            "professionalism": ProfessionalismRater()
        }
        analyzer = TraitAnalyzer(scorers=scorers)

        # Dummy data
        profile_data = {"handle": "test.bsky.social"}
        posts = [{"text": "This is a very professional post about my art."}]

        # Run analysis
        scores = await analyzer.analyze(profile_data, posts)

        # Assertions
        self.assertIn("aesthetics", scores)
        self.assertIn("professionalism", scores)
        self.assertIsInstance(scores["aesthetics"], float)
        self.assertEqual(scores["professionalism"], 0.8) # 8 / 10.0

        # Verify the LLM was called
        mock_run_llm.assert_called_once()
        call_args = mock_run_llm.call_args[0][0]
        self.assertIn("Assess the professionalism score", call_args)
        self.assertIn("This is a very professional post", call_args)

    def test_talent_manager(self):
        """Test the TalentManager's profile saving, loading, and listing."""
        manager = TalentManager(db_file=self.test_db_file)

        # 1. Test saving a profile
        profile_1 = {
            'anonymized_id': 'anon_123',
            'handle': 'test_user_1',
            'platform': 'test_platform',
            'display_name': 'Test User One',
            'scores': {'aesthetics': 0.8}
        }
        result = manager.save_profile(profile_1)
        self.assertIn("Successfully saved profile", result)

        # 2. Test retrieving the profile
        retrieved_profile = manager.get_profile('anon_123')
        self.assertIsNotNone(retrieved_profile)
        self.assertEqual(retrieved_profile['handle'], 'test_user_1')
        self.assertIn('last_saved_at', retrieved_profile)

        # 3. Test listing profiles
        profile_2 = {
            'anonymized_id': 'anon_456',
            'handle': 'test_user_2',
            'platform': 'test_platform',
            'display_name': 'Test User Two'
        }
        manager.save_profile(profile_2)

        profile_list = manager.list_profiles()
        self.assertEqual(len(profile_list), 2)
        handles = {p['handle'] for p in profile_list}
        self.assertEqual(handles, {'test_user_1', 'test_user_2'})

        # 4. Verify persistence by creating a new manager instance
        new_manager = TalentManager(db_file=self.test_db_file)
        self.assertEqual(len(new_manager.list_profiles()), 2)
        retrieved_again = new_manager.get_profile('anon_123')
        self.assertEqual(retrieved_again['scores']['aesthetics'], 0.8)

if __name__ == '__main__':
    unittest.main()