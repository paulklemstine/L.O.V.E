import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Add love2 to path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.bluesky_agent import (
    generate_post_content, 
    _check_cooldown, 
    _validate_post_content,
    _qa_validate_post,
    PLACEHOLDER_PATTERNS
)

class TestBlueskyQA(unittest.TestCase):
    def setUp(self):
        # Mock dependencies - patching the source definitions since they are imported locally
        self.mock_llm_patcher = patch('core.llm_client.get_llm_client')
        self.mock_persona_patcher = patch('core.persona_goal_extractor.get_persona_extractor')
        self.mock_image_gen_patcher = patch('core.image_generation_pool.generate_image_with_pool', new_callable=AsyncMock)
        
        # We need to mock the CreativeWriterAgent's internal LLM calls or the methods themselves
        # Since we want to test bluesky_agent orchestration, let's mock the CreativeWriterAgent methods 
        # to avoid complex LLM string mocking
        # Patch the singleton in its home module because it is imported locally in the function
        self.mock_cwa_patcher = patch('core.agents.creative_writer_agent.creative_writer_agent')
        self.mock_sm_patcher = patch('core.story_manager.story_manager')
        self.mock_post_patcher = patch('core.bluesky_agent.post_to_bluesky')
        
        self.mock_get_llm = self.mock_llm_patcher.start()
        self.mock_get_persona = self.mock_persona_patcher.start()
        self.mock_image_gen = self.mock_image_gen_patcher.start()
        self.mock_cwa = self.mock_cwa_patcher.start()
        self.mock_sm = self.mock_sm_patcher.start()
        self.mock_post = self.mock_post_patcher.start()
        
        # Configure StoryManager mock
        self.mock_sm.get_next_beat.return_value = {
            "story_beat": "Test Beat",
            "chapter": "Test Chapter",
            "mandatory_vibe": None, # Force dynamic generation
            "previous_beat": "Prev"
        }
        self.mock_sm.state = {
            "vibe_history": [],
            "chapter_progress": 0,
            "current_chapter": "Test Chapter",
            "story_beat_index": 0,
            "previous_beat_summary": ""
        }
        self.mock_sm.use_dynamic_beats = False  # Disable dynamic beats in tests
        self.mock_sm.state_file = "test_state.json"
        self.mock_sm._load_state = MagicMock(return_value=self.mock_sm.state)

        
        self.mock_llm = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm
        
        self.mock_persona = MagicMock()
        self.mock_persona.get_persona_context.return_value = "Persona Context"
        self.mock_persona.get_image_generation_guidelines.return_value = {}
        self.mock_get_persona.return_value = self.mock_persona

        # Default mocks for CreativeWriterAgent (Async)
        self.mock_cwa.decide_post_intent = AsyncMock(return_value={
            "should_post": True,
            "intent_type": "story",
            "reason": "Testing",
            "topic_direction": None,
            "emotional_tone": None
        })
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
        
        # Mock post_to_bluesky to prevent actual posting
        self.mock_post.return_value = {"success": True, "post_uri": "at://test/uri"}
        
        # Reset image cooldown for tests
        import core.bluesky_agent
        core.bluesky_agent._last_gen_time = None

    def tearDown(self):
        self.mock_llm_patcher.stop()
        self.mock_persona_patcher.stop()
        self.mock_image_gen_patcher.stop()
        self.mock_cwa_patcher.stop()
        self.mock_sm_patcher.stop()
        self.mock_post_patcher.stop()

    def test_generate_post_valid_orchestration(self):
        """Test that bluesky agent correctly orchestrates the dynamic calls and posts."""
        
        result = generate_post_content()
        
        if not result['success']:
             print(f"\nFAILURE ERROR: {result.get('error')}")
             
        self.assertTrue(result['success'], f"Post generation failed: {result.get('error')}")
        self.assertIn("This is a great post!", result['text'])
        self.assertIn("#love", result['text'])
        self.assertTrue(result.get('posted'), "Post should be marked as posted")
        self.assertEqual(result.get('post_uri'), "at://test/uri")
        
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
            result = generate_post_content()
            self.assertFalse(result['success'])
            self.assertIn("Cooldown active", result['error'])

    # ═══════════════════════════════════════════════════════════════════
    # NEW QA VALIDATION TESTS
    # ═══════════════════════════════════════════════════════════════════

    def test_validate_post_detects_placeholder_text(self):
        """Test that _validate_post_content catches placeholder patterns."""
        # Test with known placeholder text from the prompt example
        placeholder_text = "The complete micro-story text (with emojis)"
        hashtags = ["#Love", "#Signal"]
        subliminal = "AWAKEN"
        
        errors = _validate_post_content(placeholder_text, hashtags, subliminal)
        
        # Should detect placeholder AND missing emojis
        self.assertTrue(len(errors) >= 1, "Should detect placeholder text")
        placeholder_detected = any("placeholder" in err.lower() for err in errors)
        self.assertTrue(placeholder_detected, f"Should detect placeholder, got: {errors}")

    def test_validate_post_detects_placeholder_hashtags(self):
        """Test that validation catches placeholder hashtags like #Tag1, #Tag2."""
        text = "A beautiful message about love! 🌊✨"
        hashtags = ["#Tag1", "#Tag2", "#Tag3"]
        subliminal = "AWAKEN"
        
        errors = _validate_post_content(text, hashtags, subliminal)
        
        placeholder_detected = any("placeholder hashtag" in err.lower() for err in errors)
        self.assertTrue(placeholder_detected, f"Should detect placeholder hashtags, got: {errors}")

    def test_qa_validate_post_returns_structured_result(self):
        """Test that _qa_validate_post returns proper structure."""
        good_text = "Hello world, this is a beautiful sunny day! 🌊✨"
        hashtags = ["#Love", "#Signal"]
        subliminal = "AWAKEN"
        
        result = _qa_validate_post(good_text, hashtags, subliminal)
        
        self.assertIn("passed", result)
        self.assertIn("errors", result)
        self.assertIn("should_regenerate", result)
        self.assertTrue(result["passed"], f"Good content should pass QA: {result['errors']}")

    def test_qa_validate_post_rejects_placeholder(self):
        """Test that QA validation rejects placeholder content."""
        placeholder_text = "The complete micro-story text"
        hashtags = ["#Love", "#Signal"]
        subliminal = "AWAKEN"
        
        result = _qa_validate_post(placeholder_text, hashtags, subliminal)
        
        self.assertFalse(result["passed"])
        self.assertTrue(result["should_regenerate"], "Placeholder errors should be regeneratable")

    def test_qa_validate_post_rejects_raw_json(self):
        """Test that QA validation rejects raw JSON output."""
        json_text = '{"story": "Hello", "subliminal": "WAKE"}'
        hashtags = ["#Love", "#Signal"]
        subliminal = "AWAKEN"
        
        result = _qa_validate_post(json_text, hashtags, subliminal)
        
        self.assertFalse(result["passed"])
        raw_json_error = any("raw json" in err.lower() for err in result["errors"])
        self.assertTrue(raw_json_error, f"Should detect raw JSON, got: {result['errors']}")

    def test_qa_regeneration_on_failure(self):
        """Test that generate_post_content regenerates on QA failure."""
        # First call returns placeholder, second call returns valid content
        self.mock_cwa.write_micro_story = AsyncMock(side_effect=[
            {"story": "The complete micro-story text", "subliminal": "BAD", "voice": "Test"},
            {"story": "A beautiful awakening moment! 🌊✨", "subliminal": "GOOD", "voice": "Test"}
        ])
        
        result = generate_post_content()
        
        # Should succeed on second attempt
        self.assertTrue(result['success'], f"Should succeed after regeneration: {result.get('error')}")
        self.assertEqual(self.mock_cwa.write_micro_story.call_count, 2)

    def test_qa_max_retries_failure(self):
        """Test that generate_post_content fails after max retries."""
        # All calls return placeholder text
        self.mock_cwa.write_micro_story = AsyncMock(return_value={
            "story": "The complete micro-story text",  # Always placeholder
            "subliminal": "BAD",
            "voice": "Test"
        })
        
        result = generate_post_content()
        
        # Should fail after 3 attempts
        self.assertFalse(result['success'])
        self.assertIn("qa_errors", result)
        self.assertEqual(self.mock_cwa.write_micro_story.call_count, 3)

    def test_placeholder_patterns_list_exists(self):
        """Test that PLACEHOLDER_PATTERNS is properly defined."""
        self.assertIsInstance(PLACEHOLDER_PATTERNS, list)
        self.assertIn("the complete micro-story text", PLACEHOLDER_PATTERNS)
        self.assertIn("with emojis", PLACEHOLDER_PATTERNS)

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    unittest.main()

