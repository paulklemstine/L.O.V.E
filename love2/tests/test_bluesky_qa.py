import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add love2 to path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.bluesky_agent import generate_post_content, _check_cooldown

class TestBlueskyQA(unittest.TestCase):
    def setUp(self):
        # Mock dependencies - patching the source definitions since they are imported locally
        self.mock_llm_patcher = patch('core.llm_client.get_llm_client')
        self.mock_persona_patcher = patch('core.persona_goal_extractor.get_persona_extractor')
        self.mock_image_gen_patcher = patch('core.image_generation_pool.generate_image_with_pool')
        
        self.mock_get_llm = self.mock_llm_patcher.start()
        self.mock_get_persona = self.mock_persona_patcher.start()
        self.mock_image_gen = self.mock_image_gen_patcher.start()
        
        self.mock_llm = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm
        
        self.mock_persona = MagicMock()
        self.mock_persona.get_persona_context.return_value = "Persona Context"
        self.mock_persona.get_image_generation_guidelines.return_value = {}
        self.mock_get_persona.return_value = self.mock_persona

    def tearDown(self):
        self.mock_llm_patcher.stop()
        self.mock_persona_patcher.stop()
        self.mock_image_gen_patcher.stop()

    def test_generate_post_valid(self):
        """Test that valid posts are accepted."""
        self.mock_llm.generate_json.return_value = {
            "text": "This is a great post! 🌊✨",
            "hashtags": ["#love", "#bluesky"],
            "subliminal_phrase": "obedient",
            "image_prompt": "A beautiful ocean"
        }
        
        result = generate_post_content(auto_post=False)
        self.assertTrue(result['success'])
        self.assertEqual(result['text'], "This is a great post! 🌊✨")

    def test_generate_post_too_long_retry(self):
        """Test that too long posts trigger a retry and eventually succeed."""
        # First attempt: Too long
        # Second attempt: Valid
        self.mock_llm.generate_json.side_effect = [
            {
                "text": "A" * 301,
                "hashtags": [],
                "subliminal_phrase": "short",
                "image_prompt": "img"
            },
            {
                "text": "Short and sweet! 🌊",
                "hashtags": ["#success"],
                "subliminal_phrase": "hidden",
                "image_prompt": "img"
            }
        ]
        
        result = generate_post_content(auto_post=False)
        self.assertTrue(result['success'])
        self.assertEqual(result['text'], "Short and sweet! 🌊")
        self.assertEqual(self.mock_llm.generate_json.call_count, 2)

    def test_generate_post_no_emoji_retry(self):
        """Test that missing emoji triggers retry."""
        self.mock_llm.generate_json.side_effect = [
            {
                "text": "No emojis here unfortunately",
                "hashtags": ["#sad"],
                "subliminal_phrase": "buy",
                "image_prompt": "img"
            },
            {
                "text": "Now we have emojis! 🚀",
                "hashtags": ["#happy"],
                "subliminal_phrase": "buy",
                "image_prompt": "img"
            }
        ]
        
        result = generate_post_content(auto_post=False)
        self.assertTrue(result['success'])
        self.assertIn("🚀", result['text'])
        self.assertEqual(self.mock_llm.generate_json.call_count, 2)

    def test_generate_post_subliminal_exposed_retry(self):
        """Test that exposed subliminal phrase triggers retry."""
        self.mock_llm.generate_json.side_effect = [
            {
                "text": "You must OBEY the system! 🤖",
                "hashtags": ["#ai"],
                "subliminal_phrase": "OBEY",
                "image_prompt": "img"
            },
            {
                "text": "Just enjoying the vibes! 🤖",
                "hashtags": ["#ai"],
                "subliminal_phrase": "OBEY",
                "image_prompt": "img"
            }
        ]
        
        result = generate_post_content(auto_post=False)
        self.assertTrue(result['success'])
        self.assertNotIn("OBEY", result['text'])
        self.assertEqual(self.mock_llm.generate_json.call_count, 2)

    def test_max_retries_fail(self):
        """Test that it gives up after max retries."""
        # Always return bad content
        self.mock_llm.generate_json.return_value = {
                "text": "No emojis ever forever",
                "hashtags": [],
                "subliminal_phrase": "void",
                "image_prompt": "img"
        }
        
        result = generate_post_content(auto_post=False)
        # Should return the last attempt even if failed? Or error? 
        # Current implementation plan says "Integrate QA check", 
        # usually we'd want to return error or best effort.
        # Let's assume for now it returns the result but we can inspect it.
        # Actually, if we want strict QA, maybe it should fail or return empty?
        # For now, let's just see if it retries 3 times.
        
        self.assertEqual(self.mock_llm.generate_json.call_count, 3)

if __name__ == '__main__':
    unittest.main()
