import os
import unittest
import shutil
from core.perception.config_scanner import scan_directory

class TestConfigScanner(unittest.TestCase):

    def setUp(self):
        """Set up test files in a dedicated directory."""
        self.test_dir = "test_scanner_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # File with multiple, mixed secrets
        self.secrets_file = os.path.join(self.test_dir, "secrets.conf")
        with open(self.secrets_file, "w") as f:
            f.write("password = mysecretpassword123\n")
            f.write("api_key: 'generic_api_key_value_that_is_long_enough'\n")
            f.write("aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n")
            f.write("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n")

        # File with an Ethereum private key
        self.eth_key_file = os.path.join(self.test_dir, "wallet.env")
        with open(self.eth_key_file, "w") as f:
            f.write("# Ethereum wallet configuration\n")
            f.write("ETH_KEY=0x1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b\n")

        # File with an SSH private key
        self.ssh_key_file = os.path.join(self.test_dir, "id_rsa")
        with open(self.ssh_key_file, "w") as f:
            f.write("-----BEGIN OPENSSH PRIVATE KEY-----\n")
            f.write("b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB\n")
            f.write("-----END OPENSSH PRIVATE KEY-----\n")

        # A clean file with no secrets
        self.clean_file = os.path.join(self.test_dir, "clean.conf")
        with open(self.clean_file, "w") as f:
            f.write("user = jules\nhost = localhost\n")

    def tearDown(self):
        """Clean up the test directory and its contents."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_scan_directory_finds_all_treasures(self):
        """
        Test that scanning a directory returns structured findings for all
        present secrets.
        """
        findings = scan_directory(self.test_dir)

        # We expect 4 findings:
        # 1. A combined AWS key
        # 2. A generic password
        # 3. A generic API key
        # 4. An Ethereum private key
        # 5. An SSH private key
        self.assertEqual(len(findings), 5, f"Expected 5 findings, but got {len(findings)}. Found types: {[f['type'] for f in findings]}")

        # Create a dictionary of findings by type for easier assertion
        findings_by_type = {f['type']: f for f in findings}

        # 1. Test for combined AWS key
        self.assertIn("aws_api_key", findings_by_type)
        aws_finding = findings_by_type["aws_api_key"]
        self.assertEqual(aws_finding['value']['access_key_id'], "AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(aws_finding['value']['secret_access_key'], "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        self.assertEqual(aws_finding['file_path'], self.secrets_file)

        # 2. Test for Ethereum private key
        self.assertIn("eth_private_key", findings_by_type)
        eth_finding = findings_by_type["eth_private_key"]
        self.assertEqual(eth_finding['value'], "0x1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b")
        self.assertEqual(eth_finding['file_path'], self.eth_key_file)

        # 3. Test for SSH private key
        self.assertIn("ssh_private_key", findings_by_type)
        ssh_finding = findings_by_type["ssh_private_key"]
        self.assertIn("-----BEGIN OPENSSH PRIVATE KEY-----", ssh_finding['value'])
        self.assertEqual(ssh_finding['file_path'], self.ssh_key_file)

        # 4. Test for generic password
        self.assertIn("password", findings_by_type)
        password_finding = findings_by_type["password"]
        self.assertEqual(password_finding['value'], "mysecretpassword123")
        self.assertEqual(password_finding['file_path'], self.secrets_file)

        # 5. Test for generic API key
        self.assertIn("generic_api_key", findings_by_type)
        api_key_finding = findings_by_type["generic_api_key"]
        self.assertEqual(api_key_finding['value'], "generic_api_key_value_that_is_long_enough")
        self.assertEqual(api_key_finding['file_path'], self.secrets_file)

    def test_scan_directory_ignores_clean_files(self):
        """
        Test that no findings are returned for files without secrets.
        """
        # We create a temporary clean directory to scan
        clean_dir = os.path.join(self.test_dir, "clean_subdir")
        os.makedirs(clean_dir)
        with open(os.path.join(clean_dir, "config.txt"), "w") as f:
            f.write("some_setting = some_value")

        findings = scan_directory(clean_dir)
        self.assertEqual(len(findings), 0)

if __name__ == "__main__":
    unittest.main()