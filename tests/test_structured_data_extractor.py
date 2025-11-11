import unittest
from unittest.mock import patch, MagicMock
from core.structured_data_extractor import structured_content_acquisition

class TestStructuredDataExtractor(unittest.TestCase):

    @patch('requests.get')
    def test_structured_content_acquisition(self, mock_get):
        # Mock the requests.get call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <div class="item">
                    <h1>Title 1</h1>
                    <p>Description 1</p>
                </div>
                <div class="item">
                    <h1>Title 2</h1>
                    <p>Description 2</p>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        # Define a sample configuration
        configuration = {
            'sources': [
                {
                    'type': 'url',
                    'location': 'http://example.com',
                    'item_selectors': [
                        {
                            'selector': '.item',
                            'fields': {
                                'title': {'type': 'css', 'value': 'h1'},
                                'description': {'type': 'css', 'value': 'p'}
                            }
                        }
                    ]
                }
            ]
        }

        # Run the function
        results = structured_content_acquisition(configuration)

        # Assert the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], 'Title 1')
        self.assertEqual(results[0]['description'], 'Description 1')
        self.assertEqual(results[1]['title'], 'Title 2')
        self.assertEqual(results[1]['description'], 'Description 2')

if __name__ == '__main__':
    unittest.main()
