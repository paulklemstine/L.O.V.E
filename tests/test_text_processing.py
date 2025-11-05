import unittest
from unittest.mock import patch, AsyncMock
import json
from core.text_processing import process_and_structure_text

class TestProcessAndStructureText(unittest.IsolatedAsyncioTestCase):

    @patch('core.text_processing.run_llm_wrapper', new_callable=AsyncMock)
    async def test_successful_processing_from_json_string(self, mock_run_llm):
        """
        Tests that the function correctly processes a valid JSON string returned by the LLM.
        """
        mock_response = {
            "themes": ["AI development", "knowledge management"],
            "entities": [{"name": "L.O.V.E.", "type": "AI", "description": "An AI entity."}],
            "relationships": []
        }
        mock_run_llm.return_value = json.dumps(mock_response)

        raw_text = "This is a test about the L.O.V.E. AI."
        result = await process_and_structure_text(raw_text, "test_source")

        self.assertEqual(result, mock_response)
        mock_run_llm.assert_called_once()

    @patch('core.text_processing.run_llm_wrapper', new_callable=AsyncMock)
    async def test_processing_from_markdown_json(self, mock_run_llm):
        """
        Tests that the function can handle JSON wrapped in markdown code blocks.
        """
        mock_response = {"themes": ["testing"], "entities": [], "relationships": []}
        mock_llm_output = f"```json\n{json.dumps(mock_response)}\n```"
        mock_run_llm.return_value = mock_llm_output

        result = await process_and_structure_text("some text", "test_source")
        self.assertEqual(result, mock_response)

    @patch('core.text_processing.run_llm_wrapper', new_callable=AsyncMock)
    async def test_processing_from_dict(self, mock_run_llm):
        """
        Tests that the function correctly handles an already-parsed dictionary from the LLM.
        """
        mock_response = {"themes": ["direct dictionary"], "entities": [], "relationships": []}
        mock_run_llm.return_value = mock_response

        result = await process_and_structure_text("some text", "test_source")
        self.assertEqual(result, mock_response)

    @patch('core.text_processing.run_llm_wrapper', new_callable=AsyncMock)
    async def test_json_decode_error(self, mock_run_llm):
        """
        Tests the function's error handling when the LLM returns an invalid JSON string.
        """
        invalid_json = "this is not valid json"
        mock_run_llm.return_value = invalid_json

        result = await process_and_structure_text("some text", "test_source")

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to decode LLM output as JSON.")
        self.assertEqual(result["raw_output"], invalid_json)

    @patch('core.text_processing.run_llm_wrapper', new_callable=AsyncMock)
    async def test_unexpected_return_type(self, mock_run_llm):
        """
        Tests the function's error handling for unexpected data types from the LLM.
        """
        unexpected_output = 12345
        mock_run_llm.return_value = unexpected_output

        result = await process_and_structure_text("some text", "test_source")

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to process text. Unexpected data type from LLM.")
        self.assertEqual(result["raw_output"], str(unexpected_output))

if __name__ == '__main__':
    unittest.main()
