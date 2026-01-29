import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Add love2 to path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.bluesky_agent import generate_post_content, _check_cooldown

class TestBlueskyQA(unittest.TestCase):
    def setUp(self):
        # Mock dependencies - patching the source definitions since they are imported locally
        self.mock_llm_patcher = patch('core.llm_client.get_llm_client')
        self.mock_persona_patcher = patch('core.persona_goal_extractor.get_persona_extractor')
        self.mock_image_gen_patcher = patch('core.image_generation_pool.generate_image_with_pool')
        
        # We need to mock the CreativeWriterAgent's internal LLM calls or the methods themselves
        # Since we want to test bluesky_agent orchestration, let's mock the CreativeWriterAgent methods 
        # to avoid complex LLM string mocking
        # Patch the singleton in its home module because it is imported locally in the function
        self.mock_cwa_patcher = patch('core.agents.creative_writer_agent.creative_writer_agent')
        self.mock_sm_patcher = patch('core.story_manager.story_manager')
        
        self.mock_get_llm = self.mock_llm_patcher.start()
        self.mock_get_persona = self.mock_persona_patcher.start()
        self.mock_image_gen = self.mock_image_gen_patcher.start()
        self.mock_cwa = self.mock_cwa_patcher.start()
        self.mock_sm = self.mock_sm_patcher.start()
        
        # Configure StoryManager mock
        self.mock_sm.get_next_beat.return_value = {
            "story_beat": "Test Beat",
            "chapter": "Test Chapter",
            "mandatory_vibe": None, # Force dynamic generation
            "previous_beat": "Prev"
        }
        self.mock_sm.state = {"vibe_history": []}

        
        self.mock_llm = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm
        
        self.mock_persona = MagicMock()
        self.mock_persona.get_persona_context.return_value = "Persona Context"
        self.mock_persona.get_image_generation_guidelines.return_value = {}
        self.mock_get_persona.return_value = self.mock_persona

        # Default mocks for CreativeWriterAgent (Async)
        self.mock_cwa.generate_vibe = AsyncMock(return_value="Neon Bliss")
        self.mock_cwa.generate_visual_prompt = AsyncMock(return_value="A beautiful neon beach")
        self.mock_cwa.write_micro_story = AsyncMock(return_value={
            "story": "This is a great post! 🌊✨",
            "subliminal": "obedient",
            "voice": "Cyber Oracle"
        })
        self.mock_cwa.generate_manipulative_hashtags = AsyncMock(return_value=["#love", "#bluesky"])
        
        # Image gen mock
        from PIL import Image
        self.mock_image_gen.return_value = (Image.new('RGB', (100, 100)), "mock_provider")

    def tearDown(self):
        self.mock_llm_patcher.stop()
        self.mock_persona_patcher.stop()
        self.mock_image_gen_patcher.stop()
        self.mock_cwa_patcher.stop()
        self.mock_sm_patcher.stop()

    def test_generate_post_valid_orchestration(self):
        """Test that bluesky agent correctly orchestrates the dynamic calls."""
        
        result = generate_post_content(auto_post=False)
        
        if not result['success']:
             print(f"\nFAILURE ERROR: {result.get('error')}")
             
        self.assertTrue(result['success'], f"Post generation failed: {result.get('error')}")
        self.assertEqual(result['text'], "This is a great post! 🌊✨")
        self.assertEqual(result['subliminal'], "obedient")
        self.assertEqual(result['hashtags'], ["#love", "#bluesky"])
        
        # Verify dynamic calls were made
        self.mock_cwa.generate_vibe.assert_called_once()
        self.mock_cwa.generate_visual_prompt.assert_called_once_with(
            "Test Beat", # matches mock_sm.get_next_beat
            "Neon Bliss"
        )
        self.mock_image_gen.assert_called_once()

    def test_generate_post_cooldown(self):
        """Test cooldown check."""
        with patch('core.bluesky_agent._check_cooldown', return_value="Cooldown active"):
            result = generate_post_content(auto_post=True)
            self.assertFalse(result['success'])
            self.assertIn("Cooldown active", result['error'])

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    unittest.main()
