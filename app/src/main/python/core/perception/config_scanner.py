import os
import re
from typing import List, Dict, Any

# Enhanced patterns for more accurate categorization and validation.
# Each pattern now includes a 'treasure_type' to be used by the validator.
# The 'value_group' indicates which capture group from the regex holds the secret.
SECRET_PATTERNS = {
    # AWS: Requires special handling to pair access and secret keys.
    "AWS Access Key ID": {
        "pattern": r'(AKIA[0-9A-Z]{16})',
        "treasure_type": "aws_access_key_id",
        "value_group": 1,
    },
    "AWS Secret Access Key": {
        # Looks for the key in various assignment formats
        "pattern": r'(?i)aws_secret_access_key\s*[:=]\s*[\'"]?([a-zA-Z0-9/+=]{40})[\'"]?',
        "treasure_type": "aws_secret_access_key",
        "value_group": 1,
    },
    # Crypto Keys
    "Ethereum Private Key": {
        # Looks for a 64-char hex string assigned to a variable with a relevant name.
        "pattern": r'(?i)(private_key|eth_key|priv_key)\s*[:=]\s*[\'"]?(0x[a-fA-F0-9]{64})[\'"]?',
        "treasure_type": "eth_private_key",
        "value_group": 2, # The second group captures the key itself
    },
    # Generic Private Key Files
    "SSH Private Key": {
        "pattern": r'(-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----)',
        "treasure_type": "ssh_private_key",
        "value_group": 1, # The match itself is the indicator
    },
    # Other Service Keys
    "Google API Key": {
        "pattern": r'(AIza[0-9A-Za-z\\-_]{35})',
        "treasure_type": "google_api_key",
        "value_group": 1,
    },
    "GitHub Token": {
        "pattern": r'(ghp_[0-9a-zA-Z]{36})',
        "treasure_type": "github_token",
        "value_group": 1,
    },
    # Generic Patterns (less specific)
    "Generic API Key": {
        "pattern": r'(?i)api_key\s*[:=]\s*[\'"]?([^\'"\s]{16,})[\'"]?',
        "treasure_type": "generic_api_key",
        "value_group": 1,
    },
    "Generic Password": {
        "pattern": r'(?i)password\s*[:=]\s*[\'"]?([^\'"\s]{8,})[\'"]?',
        "treasure_type": "password",
        "value_group": 1,
    },
}

# File extensions to target for scanning
TARGET_EXTENSIONS = [
    ".conf", ".ini", ".json", ".yml", ".yaml", ".pem", ".key", ".env", ".xml", ".properties", ".toml", ".sh", ".bashrc", ".zshrc", ".profile"
]

def scan_file_for_secrets(filepath: str) -> List[Dict[str, Any]]:
    """
    Scans a single file for hardcoded secrets, returning structured findings.

    Args:
        filepath: The path to the file to scan.

    Returns:
        A list of dictionaries, where each dictionary represents a structured finding.
    """
    findings = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Temporary storage for secrets found in this file
        temp_findings = {}

        for name, details in SECRET_PATTERNS.items():
            matches = re.finditer(details["pattern"], content)
            for match in matches:
                value = match.group(details["value_group"])
                # For SSH keys, the value is the whole file content for context
                if details["treasure_type"] == "ssh_private_key":
                    value = content

                # Store all matches of each type found
                if details["treasure_type"] not in temp_findings:
                    temp_findings[details["treasure_type"]] = []
                temp_findings[details["treasure_type"]].append(value)

        # --- Special Handling for AWS Keys ---
        aws_access_key_list = temp_findings.pop("aws_access_key_id", [])
        aws_secret_key_list = temp_findings.pop("aws_secret_access_key", [])

        if aws_access_key_list and aws_secret_key_list:
            # If both parts of an AWS key are found, create a single, combined finding
            findings.append({
                "file_path": filepath,
                "type": "aws_api_key", # This type matches the validator
                "value": {
                    "access_key_id": aws_access_key_list[0],
                    "secret_access_key": aws_secret_key_list[0],
                },
                "content": content, # Include content for context
            })

        # --- Generic Handling for all other keys ---
        for treasure_type, values in temp_findings.items():
            for value in values:
                findings.append({
                    "file_path": filepath,
                    "type": treasure_type,
                    "value": value,
                    "content": content, # Include content for context
                })

    except Exception as e:
        # In a real scenario, this should be logged properly.
        print(f"Error scanning file {filepath}: {e}")

    return findings


def scan_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Recursively scans a directory for files with sensitive information.

    Args:
        directory: The path to the directory to scan.

    Returns:
        A list of all structured findings.
    """
    all_findings = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            # Check if the file has a target extension, is a dotfile with no extension, or has no extension at all
            if any(file.endswith(ext) for ext in TARGET_EXTENSIONS) or (file.startswith('.') and '.' not in file[1:]) or ('.' not in file):
                file_findings = scan_file_for_secrets(filepath)
                if file_findings:
                    all_findings.extend(file_findings)
    return all_findings