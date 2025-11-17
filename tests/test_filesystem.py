import os
import unittest
import shutil
from filesystem import analyze_filesystem

class TestFilesystem(unittest.TestCase):

    def setUp(self):
        """Set up test files in a dedicated directory."""
        self.test_dir = "/tmp/test_filesystem_data"
        os.makedirs(self.test_dir, exist_ok=True)

        self.secrets_file = os.path.join(self.test_dir, "secrets.conf")
        with open(self.secrets_file, "w") as f:
            f.write("password = mysecretpassword123\n")

    def tearDown(self):
        """Clean up the test directory and its contents."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_analyze_filesystem_finds_secrets(self):
        """
        Test that analyzing a directory returns structured findings for
        present secrets.
        """
        result = analyze_filesystem(self.test_dir)
        self.assertNotIn("error", result)
        self.assertEqual(len(result["validated_treasures"]), 1)
        self.assertEqual(result["validated_treasures"][0]['type'], "password")

    def test_analyze_filesystem_ignores_agent_root(self):
        """
        Test that analyzing the agent's own root directory is blocked.
        """
        result = analyze_filesystem('.')
        self.assertIn("error", result)
        self.assertEqual(len(result["validated_treasures"]), 0)

if __name__ == "__main__":
    unittest.main()
