import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import asyncio
from core.metaphor_generator import MetaphorGenerator

class TestMetaphorGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = MetaphorGenerator()

    @patch('core.metaphor_generator.run_llm', new_callable=AsyncMock)
    def test_generate_metaphor_success(self, mock_run_llm):
        # Mock successful LLM response
        mock_response_data = {
            "concept_analysis": {
                "core_attributes": ["complexity", "connection"],
                "associated_feelings": ["awe", "confusion"],
                "structural_relationships": ["nodes", "edges"]
            },
            "metaphors": [
                {
                    "id": 1,
                    "textual_description": "A vast web...",
                    "visual_prompt": "Spider web glowing..."
                },
                {
                    "id": 2,
                    "textual_description": "A root system...",
                    "visual_prompt": "Roots intertwining..."
                },
                {
                    "id": 3,
                    "textual_description": "A starry sky...",
                    "visual_prompt": "Constellations connected..."
                }
            ]
        }
        mock_run_llm.return_value = {"result": json.dumps(mock_response_data)}

        # Run the generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.generator.generate_metaphor("complexity", "scientific"))
        loop.close()

        # Verify results
        self.assertEqual(result, mock_response_data)
        mock_run_llm.assert_called_once()
        call_args = mock_run_llm.call_args
        self.assertIn("complexity", call_args.kwargs['prompt_text'])
        self.assertIn("scientific", call_args.kwargs['prompt_text'])
        self.assertEqual(call_args.kwargs['purpose'], "creative_writing")

    @patch('core.metaphor_generator.run_llm', new_callable=AsyncMock)
    def test_generate_metaphor_json_error(self, mock_run_llm):
        # Mock invalid JSON response
        mock_run_llm.return_value = {"result": "Not valid JSON"}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.generator.generate_metaphor("chaos", "dark"))
        loop.close()

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid JSON response from LLM")

    @patch('core.metaphor_generator.run_llm', new_callable=AsyncMock)
    def test_generate_metaphor_empty_response(self, mock_run_llm):
        # Mock empty response
        mock_run_llm.return_value = {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.generator.generate_metaphor("nothing", "neutral"))
        loop.close()

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to generate metaphors.")

if __name__ == '__main__':
    unittest.main()
