import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_api import get_openrouter_models

class TestOpenRouterFreeModels(unittest.TestCase):

    @patch('core.llm_api.requests.get')
    @patch('core.llm_api.os.environ.get')
    def test_get_openrouter_models_logic(self, mock_env, mock_get):
        # Setup API Key
        mock_env.return_value = "fake_key"

        # Mock API Response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                # Valid Free Model (No Suffix) -> Should get :free
                {
                    "id": "valid/model-1",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 4096
                },
                # Valid Free Model (With Suffix) -> Should stay same
                {
                    "id": "valid/model-2:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 8192
                },
                # Paid Model -> Should be removed
                {
                    "id": "paid/model",
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                    "context_length": 4096
                },
                # OpenRouter Prefix Model (Free) -> Should be removed (User Rule)
                {
                    "id": "openrouter/auto",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "context_length": 4096
                },
                # OpenRouter Prefix Model (Paid) -> Should be removed
                {
                    "id": "openrouter/turbo",
                    "pricing": {"prompt": "1", "completion": "1"},
                    "context_length": 4096
                }
            ]
        }
        mock_get.return_value = mock_response

        # Run function
        models = get_openrouter_models()

        # Assertions
        print(f"Resulting Models: {models}")

        self.assertIn("valid/model-1:free", models)
        self.assertIn("valid/model-2:free", models)
        self.assertNotIn("paid/model", models)
        self.assertNotIn("paid/model:free", models)
        self.assertNotIn("openrouter/auto", models)
        self.assertNotIn("openrouter/auto:free", models)
        self.assertNotIn("openrouter/turbo", models)
        
        # Ensure ONLY the valid ones are there
        self.assertEqual(len(models), 2)

if __name__ == '__main__':
    unittest.main()
