import os
import unittest
from core.perception.config_scanner import scan_file_for_secrets, scan_file_for_insecure_settings, scan_directory

class TestConfigScanner(unittest.TestCase):

    def setUp(self):
        """Set up test files."""
        self.test_dir = "test_scanner_data"
        os.makedirs(self.test_dir, exist_ok=True)

        self.secrets_file = os.path.join(self.test_dir, "secrets.conf")
        with open(self.secrets_file, "w") as f:
            f.write("password=mysecretpassword\n")
            f.write("api_key: 'supersecretapikey'\n")
            f.write("AWS_SECRET_KEY = \"anothersecretkey\"\n")

        self.insecure_file = os.path.join(self.test_dir, "insecure.conf")
        with open(self.insecure_file, "w") as f:
            f.write("debug = true\n")
            f.write("PermitRootLogin yes\n")

        self.clean_file = os.path.join(self.test_dir, "clean.conf")
        with open(self.clean_file, "w") as f:
            f.write("user = jules\n")
            f.write("host = localhost\n")

        self.private_key_file = os.path.join(self.test_dir, "test.pem")
        with open(self.private_key_file, "w") as f:
            f.write("-----BEGIN RSA PRIVATE KEY-----\n")
            f.write("notarealkey\n")
            f.write("-----END RSA PRIVATE KEY-----\n")

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_scan_file_for_secrets(self):
        """Test scanning a single file for secrets."""
        findings = scan_file_for_secrets(self.secrets_file)
        self.assertEqual(len(findings), 3)
        self.assertEqual(findings[0][1], "contains_password")
        self.assertEqual(findings[0][2], "mysecretpassword")
        self.assertEqual(findings[1][1], "contains_api_key")
        self.assertEqual(findings[1][2], "supersecretapikey")
        self.assertEqual(findings[2][1], "contains_aws_secret_key")
        self.assertEqual(findings[2][2], "anothersecretkey")

    def test_scan_file_for_private_key(self):
        """Test scanning for a private key."""
        findings = scan_file_for_secrets(self.private_key_file)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0][1], "contains_private_key")
        self.assertEqual(findings[0][2], "found")

    def test_scan_file_for_insecure_settings(self):
        """Test scanning a single file for insecure settings."""
        findings = scan_file_for_insecure_settings(self.insecure_file)
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0][1], "insecure_setting_debug_enabled")
        self.assertEqual(findings[1][1], "insecure_setting_remote_login_enabled")

    def test_scan_clean_file(self):
        """Test scanning a clean file."""
        secret_findings = scan_file_for_secrets(self.clean_file)
        insecure_findings = scan_file_for_insecure_settings(self.clean_file)
        self.assertEqual(len(secret_findings), 0)
        self.assertEqual(len(insecure_findings), 0)

    def test_scan_directory(self):
        """Test scanning a directory."""
        findings = scan_directory(self.test_dir)
        self.assertEqual(len(findings), 6) # 3 secrets + 1 private key + 2 insecure settings

if __name__ == "__main__":
    unittest.main()