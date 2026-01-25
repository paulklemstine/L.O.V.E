import unittest
from unittest.mock import MagicMock
from core.knowledge_extractor import KnowledgeExtractor

class TestKnowledgeExtractor(unittest.TestCase):

    def setUp(self):
        self.mock_kb = MagicMock()
        self.extractor = KnowledgeExtractor(self.mock_kb)

    def test_parse_probe_data(self):
        self.mock_kb.get_node.return_value = None  # Simulate node not existing
        host_ip = "192.168.1.1"
        ports_data = {
            80: {'service': 'http', 'version': 'Apache/2.4.29'},
            443: {'service': 'https', 'version': 'nginx/1.14.0'}
        }
        self.extractor.parse_probe_data(host_ip, ports_data)

        # Verify that add_node and add_edge were called with the correct arguments
        self.mock_kb.get_node.assert_any_call("http_Apache/2.4.29")
        self.mock_kb.add_node.assert_any_call("http_Apache/2.4.29", 'software', attributes={'service': 'http', 'version': 'Apache/2.4.29'})
        self.mock_kb.add_edge.assert_any_call(host_ip, "http_Apache/2.4.29", 'runs_software', attributes={'port': 80})

        self.mock_kb.get_node.assert_any_call("https_nginx/1.14.0")
        self.mock_kb.add_node.assert_any_call("https_nginx/1.14.0", 'software', attributes={'service': 'https', 'version': 'nginx/1.14.0'})
        self.mock_kb.add_edge.assert_any_call(host_ip, "https_nginx/1.14.0", 'runs_software', attributes={'port': 443})

    def test_parse_web_content(self):
        url = "http://example.com"
        html_content = """
        <html>
            <head>
                <title>Example Page</title>
            </head>
            <body>
                <a href="http://example.com/page1">Page 1</a>
                <a href="/page2">Page 2 (relative)</a>
                <a href="http://google.com">Google</a>
            </body>
        </html>
        """
        self.extractor.parse_web_content(url, html_content)

        # Verify that the title was extracted and the node updated
        self.mock_kb.add_node.assert_any_call(url, 'webrequest', attributes={'title': 'Example Page'})

        # Verify that the external links were added and relationships created
        self.mock_kb.add_node.assert_any_call("http://example.com/page1", 'webrequest', attributes={})
        self.mock_kb.add_edge.assert_any_call(url, "http://example.com/page1", 'links_to')

        self.mock_kb.add_node.assert_any_call("http://google.com", 'webrequest', attributes={})
        self.mock_kb.add_edge.assert_any_call(url, "http://google.com", 'links_to')

if __name__ == '__main__':
    unittest.main()
