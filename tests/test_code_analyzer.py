import unittest
import os
from unittest.mock import patch
from io import StringIO

from code_analyzer import code_analyzer
from extract_todos import extract_todos

class TestCodeAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_file = "test_file.txt"
        with open(self.test_file, "w") as f:
            f.write("# This is a test file\n")
            f.write(" # TODO: Fix this bug\n")
            f.write("description = 'A test description'\n")
            f.write(" # another line\n")
            f.write(" # todo: another todo\n")

    def tearDown(self):
        os.remove(self.test_file)

    def test_code_analyzer(self):
        findings = code_analyzer(self.test_file)
        self.assertEqual(len(findings), 3)
        self.assertEqual(findings[0]['line_number'], 2)
        self.assertEqual(findings[0]['keyword'], 'TODO')
        self.assertEqual(findings[0]['line'], '# TODO: Fix this bug')
        self.assertEqual(findings[1]['line_number'], 3)
        self.assertEqual(findings[1]['keyword'], 'description')
        self.assertEqual(findings[1]['line'], "description = 'A test description'")
        self.assertEqual(findings[2]['line_number'], 5)
        self.assertEqual(findings[2]['keyword'], 'TODO')

    @patch('sys.stdout', new_callable=StringIO)
    def test_extract_todos(self, mock_stdout):
        extract_todos(self.test_file)
        output = mock_stdout.getvalue()
        self.assertIn("Found 2 TODOs", output)
        self.assertIn("Line 2: # TODO: Fix this bug", output)
        self.assertIn("Line 5: # todo: another todo", output)

if __name__ == '__main__':
    unittest.main()
