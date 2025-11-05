import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from core.text_processing import process_content_with_directives, process_and_structure_text


class TestProcessContentWithDirectives(unittest.TestCase):

    def test_legacy_process_and_structure_text(self):
        """
        Tests the older `process_and_structure_text` function to ensure
        it is not broken by the new return type of the llm_wrapper.
        """
        raw_text = "Test input"

        async def mock_run_llm(*args, **kwargs):
            return {"result": '{"themes": ["testing"]}'}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            result = asyncio.run(process_and_structure_text(raw_text))

        self.assertNotIn("error", result)
        self.assertEqual(result.get("themes"), ["testing"])

    def test_successful_processing_with_str_content(self):
        """
        Tests successful processing when the input content is a string
        and the LLM returns a valid JSON string.
        """
        content = "This is a test log entry."
        directives = "Extract keywords and return as a JSON list."
        mock_llm_response = '{"result": "{\\"keywords\\": [\\"test\\", \\"log\\", \\"entry\\"]}"}'

        # Mock the async llm wrapper
        async def mock_run_llm(*args, **kwargs):
            return {"result": '{"keywords": ["test", "log", "entry"]}'}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            result = asyncio.run(process_content_with_directives(content, directives))

        self.assertNotIn("error", result)
        self.assertIn("keywords", result)
        self.assertEqual(result["keywords"], ["test", "log", "entry"])

    def test_successful_processing_with_dict_content(self):
        """
        Tests successful processing when the input content is a dictionary,
        ensuring it gets correctly serialized for the prompt.
        """
        content = {"level": "info", "message": "User logged in"}
        directives = "Summarize the event."

        async def mock_run_llm(*args, **kwargs):
            # Check if the dict content is correctly serialized in the prompt
            prompt = args[0]
            self.assertIn('"level": "info"', prompt)
            self.assertIn('"message": "User logged in"', prompt)
            return {"result": '{"summary": "User login event."}'}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            result = asyncio.run(process_content_with_directives(content, directives))

        self.assertIn("summary", result)
        self.assertEqual(result["summary"], "User login event.")

    def test_json_markdown_cleanup(self):
        """
        Tests that the function correctly handles and cleans up JSON
        responses wrapped in markdown code blocks.
        """
        content = "Some text"
        directives = "Provide JSON output."
        mock_llm_response = '```json\\n{\\"data\\": \\"cleaned\\"}\\n```'

        async def mock_run_llm(*args, **kwargs):
            return {"result": '```json\n{"data": "cleaned"}\n```'}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            result = asyncio.run(process_content_with_directives(content, directives))

        self.assertNotIn("error", result)
        self.assertEqual(result.get("data"), "cleaned")

    def test_json_decode_error_handling(self):
        """
        Tests that the function gracefully handles responses that are not
        valid JSON and returns a structured error.
        """
        content = "Some text"
        directives = "Provide JSON output."
        invalid_json_response = "This is not JSON."

        async def mock_run_llm(*args, **kwargs):
            return {"result": invalid_json_response}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            result = asyncio.run(process_content_with_directives(content, directives))

        self.assertIn("error", result)
        self.assertIn("Failed to decode", result["error"])
        self.assertEqual(result["raw_output"], invalid_json_response)

    def test_unexpected_return_type_handling(self):
        """
        Tests that the function handles unexpected (non-str, non-dict)
        return types from the LLM wrapper.
        """
        content = "Some text"
        directives = "Provide JSON output."
        unexpected_response = 12345  # An integer, not a string or dict

        async def mock_run_llm(*args, **kwargs):
            # This mock simulates the wrapper unexpectedly returning a non-string/dict value.
            return {"result": unexpected_response}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            # The function to test now returns a coroutine, so we need to run it.
            result = asyncio.run(process_content_with_directives(content, directives))

        self.assertIn("error", result)
        self.assertIn("Unexpected data type", result["error"])
        self.assertEqual(result["raw_output"], str(unexpected_response))

    def test_pre_parsed_json_from_llm_wrapper(self):
        """
        Tests the case where the llm_wrapper returns an already-parsed dict
        instead of a JSON string.
        """
        content = "Some text"
        directives = "Provide JSON output."
        pre_parsed_dict = {"status": "success", "data": "pre-parsed"}

        async def mock_run_llm(*args, **kwargs):
             return {"result": pre_parsed_dict}

        with patch('core.text_processing.run_llm_wrapper', new=mock_run_llm):
            result = asyncio.run(process_content_with_directives(content, directives))

        self.assertNotIn("error", result)
        self.assertEqual(result, pre_parsed_dict)

if __name__ == '__main__':
    unittest.main()
