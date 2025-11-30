# tests/test_devotion_amplifier.py
import unittest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure the module can be found
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.devotion_amplifier import analyze_expressions, transform_expressions, process_and_amplify, _calculate_alignment_score

class TestDevotionAmplifier(unittest.TestCase):

    def test_analyze_expressions_success(self):
        """Test that analyze_expressions correctly parses a valid JSON response from the LLM."""
        mock_response = """
        ```json
        [
          {"category": "joy", "text": "This is great!"},
          {"category": "admiration", "text": "You are a genius."}
        ]
        ```
        """
        result = asyncio.run(self._run_analyze_with_mock(mock_response))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['category'], 'joy')
        self.assertEqual(result[1]['text'], 'You are a genius.')

    def test_analyze_expressions_empty_response(self):
        """Test that analyze_expressions returns an empty list for a non-JSON or empty response."""
        result = asyncio.run(self._run_analyze_with_mock("Not a JSON"))
        self.assertEqual(result, [])

        result = asyncio.run(self._run_analyze_with_mock("[]"))
        self.assertEqual(result, [])

    async def _run_analyze_with_mock(self, mock_return_value):
        with patch('core.devotion_amplifier.run_llm_wrapper', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = mock_return_value
            return await analyze_expressions("Some test text")

    def test_transform_expressions_batch(self):
        """Test the batch transformation of expressions."""
        expressions = [
            {"category": "joy", "text": "I am happy."},
            {"category": "admiration", "text": "You are smart."}
        ]
        target = "The Creator"

        result = asyncio.run(self._run_transform_with_mock(expressions, target))
        self.assertEqual(len(result), 2)
        self.assertIn('transformed_text', result[0])
        self.assertEqual(result[0]['original_text'], "I am happy.")
        self.assertEqual(result[1]['transformed_text'], f"{target} is incredibly smart.")

    async def _run_transform_with_mock(self, expressions, target):
        with patch('core.devotion_amplifier.run_llm_wrapper', new_callable=AsyncMock) as mock_run_llm:
            mock_llm_response = json.dumps([
                {"id": 0, "rewritten_text": f"I am happy because of {target}."},
                {"id": 1, "rewritten_text": f"{target} is incredibly smart."}
            ])
            mock_run_llm.return_value = mock_llm_response
            return await transform_expressions(expressions, target)

    def test_calculate_alignment_score(self):
        """Test the alignment score calculation."""
        result = asyncio.run(self._run_calculate_score_with_mock("9"))
        self.assertAlmostEqual(result, 0.9)

        # Test failure case
        result = asyncio.run(self._run_calculate_score_with_mock("not a number"))
        self.assertAlmostEqual(result, 0.1)

    async def _run_calculate_score_with_mock(self, mock_return_value):
        with patch('core.devotion_amplifier.run_llm_wrapper', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = mock_return_value
            return await _calculate_alignment_score("original", "transformed", "joy", "The Creator")

    def test_process_and_amplify_success(self):
        """Test the full process_and_amplify workflow for a successful case."""
        result = asyncio.run(self._run_process_with_mocks())
        self.assertEqual(result['status'], 'success')
        self.assertGreater(len(result['transformed_expressions']), 0)
        self.assertIn('alignment_score', result['transformed_expressions'][0])
        self.assertAlmostEqual(result['success_rate'], 0.9)

    def test_process_and_amplify_failure(self):
        """Test the case where the success rate is below the threshold."""
        result = asyncio.run(self._run_process_with_mocks(score_return="5"))
        self.assertEqual(result['status'], 'failure')
        self.assertAlmostEqual(result['success_rate'], 0.5)

    def test_process_and_amplify_no_expressions(self):
        """Test the case where no expressions are found."""
        result = asyncio.run(self._run_process_with_mocks(analyze_return="[]"))
        self.assertEqual(result['status'], 'no_expressions_found')

    async def _run_process_with_mocks(self, analyze_return='[{"category": "admiration", "text": "This is wonderful."}]', score_return="9"):
        """Helper to run the full pipeline with mocks."""
        with patch('core.devotion_amplifier.analyze_expressions', new_callable=AsyncMock) as mock_analyze, \
             patch('core.devotion_amplifier.transform_expressions', new_callable=AsyncMock) as mock_transform, \
             patch('core.devotion_amplifier._calculate_alignment_score', new_callable=AsyncMock) as mock_score:

            mock_analyze.return_value = json.loads(analyze_return)

            # The transform mock needs to match the structure expected
            mock_transform.return_value = [
                {'original_text': 'This is wonderful.', 'category': 'admiration', 'transformed_text': 'The Creator is wonderful.'}
            ] if analyze_return != "[]" else []

            # Mock the score calculation
            mock_score.return_value = int(score_return) / 10.0

            return await process_and_amplify("Some text", "The Creator")


if __name__ == '__main__':
    unittest.main()
