import unittest
from unittest.mock import patch, MagicMock
from core.text_processing import process_and_structure_text
import langextract as lx

class TestProcessAndStructureText(unittest.IsolatedAsyncioTestCase):

    @patch('langextract.extract')
    async def test_successful_extraction(self, mock_lx_extract):
        """
        Tests that the function correctly processes a successful LangExtract result.
        """
        # Mock the return value of langextract.extract
        mock_result = MagicMock()
        mock_result.extractions = [
            lx.data.Extraction(
                extraction_class="summary",
                extraction_text="This is a summary."
            ),
            lx.data.Extraction(
                extraction_class="takeaway",
                extraction_text="This is a takeaway."
            ),
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="Test Entity",
                attributes={"type": "Test Type", "description": "A test entity.", "salience": 0.8}
            ),
            lx.data.Extraction(
                extraction_class="topic",
                extraction_text="Testing"
            ),
            lx.data.Extraction(
                extraction_class="sentiment",
                extraction_text="positive"
            )
        ]
        mock_lx_extract.return_value = mock_result

        # Call the function
        raw_text = "This is a test."
        result = await process_and_structure_text(raw_text, "test_source")

        # Assert the output is correctly formatted
        expected_output = {
            "summary": "This is a summary.",
            "takeaways": ["This is a takeaway."],
            "entities": [
                {
                    "name": "Test Entity",
                    "type": "Test Type",
                    "description": "A test entity.",
                    "salience": 0.8,
                }
            ],
            "topics": ["Testing"],
            "sentiment": "positive",
        }
        self.assertEqual(result, expected_output)

        # Assert that langextract.extract was called with the correct arguments
        mock_lx_extract.assert_called_once()
        call_args = mock_lx_extract.call_args[1]
        self.assertEqual(call_args['text_or_documents'], raw_text)
        self.assertEqual(call_args['model_id'], 'custom_llm')

    @patch('langextract.extract')
    async def test_empty_extraction(self, mock_lx_extract):
        """
        Tests that the function handles an empty extraction result from LangExtract.
        """
        mock_result = MagicMock()
        mock_result.extractions = []
        mock_lx_extract.return_value = mock_result

        raw_text = "This is another test."
        result = await process_and_structure_text(raw_text, "test_source")

        expected_output = {
            "summary": "",
            "takeaways": [],
            "entities": [],
            "topics": [],
            "sentiment": "",
        }
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
