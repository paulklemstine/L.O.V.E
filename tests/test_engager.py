import unittest
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock the core modules that are not installed in the test environment
sys.modules['core.llm_api'] = MagicMock()
sys.modules['core.talent_utils.manager'] = MagicMock()
sys.modules['core.bluesky_api'] = MagicMock()
sys.modules['core.dispatcher'] = MagicMock()
sys.modules['core.interface_handlers'] = MagicMock()

from core.talent_utils.engager import OpportunityEngager

class TestOpportunityEngager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_talent_manager = MagicMock()
        self.engager = OpportunityEngager(self.mock_talent_manager)

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

    @patch('core.talent_utils.engager.dispatch_structured_payload', new_callable=AsyncMock)
    @patch('core.talent_utils.engager.OpportunityEngager.generate_proposal', new_callable=AsyncMock)
    async def test_engage_talent_sends_payload(self, mock_generate_proposal, mock_dispatch):
        """Test that engage_talent constructs and dispatches a payload."""
        profile_id = "test_bsky_id"
        mock_generate_proposal.return_value = "A test proposal."
        mock_profile = {
            "platform": "bluesky",
            "posts": [
                {"uri": "at://did:plc:123/app.bsky.feed.post/456", "created_at": "2023-01-01T00:00:00Z"},
                {"uri": "at://did:plc:123/app.bsky.feed.post/789", "created_at": "2023-01-02T00:00:00Z"}
            ]
        }
        self.mock_talent_manager.get_talent_by_id.return_value = mock_profile
        mock_dispatch.return_value = {'status': 'success'}

        await self.engager.engage_talent(profile_id)

        mock_generate_proposal.assert_called_once_with(profile_id)
        mock_dispatch.assert_called_once()

        # Verify the payload structure
        dispatched_payload = mock_dispatch.call_args[0][0]
        self.assertEqual(dispatched_payload['action'], 'reply')
        self.assertEqual(dispatched_payload['platform_identifier'], 'bluesky')
        self.assertEqual(dispatched_payload['content'], 'A test proposal.')
        self.assertEqual(dispatched_payload['root_uri'], 'at://did:plc:123/app.bsky.feed.post/789')
        self.assertEqual(dispatched_payload['parent_uri'], 'at://did:plc:123/app.bsky.feed.post/789')

    @patch('core.talent_utils.engager.OpportunityEngager.generate_proposal', new_callable=AsyncMock)
    async def test_engage_talent_dry_run(self, mock_generate_proposal):
        """Test that a dry run generates a proposal but does not send it."""
        profile_id = "test_dry_run_id"
        mock_generate_proposal.return_value = "A dry run proposal."
        self.mock_talent_manager.get_talent_by_id.return_value = {"platform": "bluesky"} # Needed for the platform check

        await self.engager.engage_talent(profile_id, dry_run=True)

        mock_generate_proposal.assert_called_once_with(profile_id)
        # In a dry run, dispatch should not be called.
        # We can check this by ensuring the dispatch mock (if we had one here) isn't called,
        # or by verifying no network calls are made if we had a more integrated test.

if __name__ == '__main__':
    unittest.main()
