import unittest
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock the core modules that are not installed in the test environment
sys.modules['core.llm_api'] = MagicMock()
sys.modules['core.talent_utils.manager'] = MagicMock()
sys.modules['core.bluesky_api'] = MagicMock()

from core.talent_utils.engager import OpportunityEngager
import core.talent_utils.engager

class TestOpportunityEngager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_talent_manager = MagicMock()
        self.engager = OpportunityEngager(self.mock_talent_manager)

        # We need to import the mocked modules *after* they have been added to sys.modules
        from core.llm_api import run_llm
        self.mock_run_llm = run_llm
        from core.bluesky_api import reply_to_post
        self.mock_reply_to_post = reply_to_post

    @patch('core.talent_utils.engager.run_llm', new_callable=AsyncMock)
    async def test_generate_proposal_success(self, mock_run_llm):
        """Test that a proposal is generated successfully for a valid profile."""
        profile_id = "test_id"
        mock_profile = {
            "handle": "test_handle",
            "display_name": "Test User",
            "bio": "A test bio.",
            "posts": [{"text": "A recent test post."}]
        }
        self.mock_talent_manager.get_talent_by_id.return_value = mock_profile
        mock_run_llm.return_value = {"result": "A beautiful proposal."}

        proposal = await self.engager.generate_proposal(profile_id)

        self.assertEqual(proposal, "A beautiful proposal.")
        self.mock_talent_manager.get_talent_by_id.assert_called_once_with(profile_id)
        mock_run_llm.assert_called_once()

    async def test_generate_proposal_profile_not_found(self):
        """Test that a specific message is returned when the profile is not found."""
        profile_id = "non_existent_id"
        self.mock_talent_manager.get_talent_by_id.return_value = None

        proposal = await self.engager.generate_proposal(profile_id)

        self.assertEqual(proposal, "Could not find a talent profile with the specified ID.")
        self.mock_talent_manager.get_talent_by_id.assert_called_once_with(profile_id)

    def test_send_proposal_to_bluesky_success(self):
        """Test that a proposal is sent successfully to a valid Bluesky profile."""
        profile_id = "test_bsky_id"
        proposal_text = "Hello, this is a test."
        mock_profile = {
            "platform": "bluesky",
            "posts": [
                {"uri": "at://did:plc:123/app.bsky.feed.post/456", "created_at": "2023-01-01T00:00:00Z"},
                {"uri": "at://did:plc:123/app.bsky.feed.post/789", "created_at": "2023-01-02T00:00:00Z"}
            ]
        }
        self.mock_talent_manager.get_talent_by_id.return_value = mock_profile
        self.mock_reply_to_post.return_value = {"uri": "some_reply_uri"}

        result = self.engager.send_proposal_to_bluesky(profile_id, proposal_text)

        self.assertTrue(result)
        self.mock_talent_manager.get_talent_by_id.assert_called_once_with(profile_id)
        # It should reply to the latest post
        latest_post_uri = "at://did:plc:123/app.bsky.feed.post/789"
        self.mock_reply_to_post.assert_called_once_with(root_uri=latest_post_uri, parent_uri=latest_post_uri, text=proposal_text)

    def test_send_proposal_to_bluesky_no_posts(self):
        """Test that sending fails if the profile has no posts."""
        profile_id = "test_bsky_id_no_posts"
        proposal_text = "Hello, this is a test."
        mock_profile = {
            "platform": "bluesky",
            "posts": []
        }
        self.mock_talent_manager.get_talent_by_id.return_value = mock_profile

        result = self.engager.send_proposal_to_bluesky(profile_id, proposal_text)

        self.assertFalse(result)
        self.mock_reply_to_post.assert_not_called()

    @patch('core.talent_utils.engager.OpportunityEngager.generate_proposal', new_callable=AsyncMock)
    @patch('core.talent_utils.engager.OpportunityEngager.send_proposal_to_bluesky')
    async def test_engage_talent_dry_run(self, mock_send_proposal, mock_generate_proposal):
        """Test that a dry run generates a proposal but does not send it."""
        profile_id = "test_dry_run_id"
        mock_generate_proposal.return_value = "A dry run proposal."
        self.mock_talent_manager.get_talent_by_id.return_value = {"platform": "bluesky"} # Needed for the platform check

        await self.engager.engage_talent(profile_id, dry_run=True)

        mock_generate_proposal.assert_called_once_with(profile_id)
        mock_send_proposal.assert_not_called()

if __name__ == '__main__':
    unittest.main()
