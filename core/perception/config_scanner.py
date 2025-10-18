import os
import re
from typing import List, Tuple

# Regular expressions for detecting sensitive information
# This is a basic set and can be expanded
SECRET_PATTERNS = {
    "password": r'(?i)password\s*[:=]\s*[\'"]?([^\'"\n]+)[\'"]?',
    "api_key": r'(?i)api_key\s*[:=]\s*[\'"]?([^\'"\n]+)[\'"]?',
    "aws_access_key": r'AKIA[0-9A-Z]{16}',
    "aws_secret_key": r'(?i)aws_secret_key\s*[:=]\s*[\'"]?([^\'"\n]+)[\'"]?',
    "google_api_key": r'AIza[0-9A-Za-z\\-_]{35}',
    "github_token": r'ghp_[0-9a-zA-Z]{36}',
    "private_key": r'-----BEGIN ((EC|RSA|DSA|OPENSSH) )?PRIVATE KEY-----',
}

# Common insecure settings and default credentials
INSECURE_SETTINGS = {
    "default_password": r'(?i)password\s*[:=]\s*[\'"]?(password|admin|root|1234)[\'"]?',
    "debug_enabled": r'(?i)debug\s*[:=]\s*true',
    "remote_login_enabled": r'(?i)PermitRootLogin\s+yes',
}

# File extensions to target for scanning
TARGET_EXTENSIONS = [
    ".conf", ".ini", ".json", ".yml", ".yaml", ".pem", ".key", ".env", ".xml", ".properties", ".toml"
]

def scan_file_for_secrets(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Scans a single file for hardcoded secrets.

    Args:
        filepath: The path to the file to scan.

    Returns:
        A list of tuples, where each tuple represents a finding in the
        form of (filepath, finding_type, finding_value).
    """
    findings = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for secret_type, pattern in SECRET_PATTERNS.items():
                matches = re.findall(pattern, content)
                for match in matches:
                    # For private keys, we don't need the value, just that it exists
                    value = match if isinstance(match, str) and match else "found"
                    findings.append((filepath, f"contains_{secret_type}", value))
    except Exception as e:
        print(f"Error scanning file {filepath}: {e}")
    return findings

def scan_file_for_insecure_settings(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Scans a single file for insecure settings.

    Args:
        filepath: The path to the file to scan.

    Returns:
        A list of tuples, where each tuple represents a finding in the
        form of (filepath, finding_type, finding_value).
    """
    findings = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for setting_type, pattern in INSECURE_SETTINGS.items():
                matches = re.findall(pattern, content)
                for match in matches:
                    value = match if isinstance(match, str) and match else "found"
                    findings.append((filepath, f"insecure_setting_{setting_type}", value))
    except Exception as e:
        print(f"Error scanning file {filepath}: {e}")
    return findings

def scan_directory(directory: str) -> List[Tuple[str, str, str]]:
    """
    Recursively scans a directory for files with sensitive information and
    insecure settings.

    Args:
        directory: The path to the directory to scan.

    Returns:
        A list of all findings.
    """
    all_findings = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if any(file.endswith(ext) for ext in TARGET_EXTENSIONS):
                findings = scan_file_for_secrets(filepath)
                if findings:
                    all_findings.extend(findings)
                findings = scan_file_for_insecure_settings(filepath)
                if findings:
                    all_findings.extend(findings)
            # Also check files without extensions, like 'sshd_config'
            elif '.' not in file:
                 findings = scan_file_for_insecure_settings(filepath)
                 if findings:
                    all_findings.extend(findings)
    return all_findings