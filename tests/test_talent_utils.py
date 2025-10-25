import unittest
import os
import hashlib
from unittest.mock import patch, Mock, MagicMock

# It's good practice to add the project root to the path for testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle
from core.talent_utils.analyzer import TraitAnalyzer, AestheticScorer, ProfessionalismRater
from core.talent_utils.manager import ContactManager

class TestTalentUtils(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Mock environment variables
        self.patcher_env = patch.dict(os.environ, {
            "BLUESKY_USER": "testuser",
            "BLUESKY_PASSWORD": "testpassword",
            "TALENT_LOG_KEY": "a_valid_fernet_key_for_testing_32_bytes"
        })
        self.patcher_env.start()

    def tearDown(self):
        """Clean up test environment after each test."""
        self.patcher_env.stop()
        if os.path.exists("contact_log.enc"):
            os.remove("contact_log.enc")

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

    @patch('core.talent_utils.analyzer.run_llm')
    def test_trait_analyzer(self, mock_run_llm):
        """Test the TraitAnalyzer with its scoring plugins."""
        # Mock the LLM response for the ProfessionalismRater
        mock_run_llm.return_value = "8"

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
        scores = analyzer.analyze(profile_data, posts)

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

    def test_contact_manager(self):
        """Test the ContactManager's message generation and encrypted logging."""
        templates = {
            'initial': "Hello [First], we admire your work in [Field]...",
        }
        constraints = {
            'max_attempts': 3,
            'min_response_window': '7 days'
        }

        # Generate a valid Fernet key for the test
        from cryptography.fernet import Fernet
        test_key = Fernet.generate_key().decode()

        # Patch the environment *before* creating the ContactManager
        with patch.dict(os.environ, {"TALENT_LOG_KEY": test_key}):
            manager = ContactManager(templates=templates, constraints=constraints)

        # Test message generation
        message = manager.generate_message("initial", {"First": "Test", "Field": "art"})
        self.assertEqual(message, "Hello Test, we admire your work in art...")

        # Test outreach recording and logging
        anonymized_id = "anon_123"
        manager.record_outreach(anonymized_id, "initial", {"First": "Test", "Field": "art"})

        # Verify log file was created and is not empty
        self.assertTrue(os.path.exists("contact_log.enc"))
        with open("contact_log.enc", "rb") as f:
            self.assertTrue(f.read())

        # Verify rate limiting
        can_contact, _ = manager.can_contact(anonymized_id)
        self.assertFalse(can_contact) # Should be false due to min_response_window

if __name__ == '__main__':
    unittest.main()