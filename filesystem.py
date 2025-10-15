import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Regex patterns for various secrets
SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret_key": re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z\\-_]{35}"),
    "github_token": re.compile(r"ghp_[0-9a-zA-Z]{36}"),
    "rsa_private_key": re.compile(r"-----BEGIN RSA PRIVATE KEY-----"),
    "ssh_private_key": re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
    "generic_api_key": re.compile(r"[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[:=]\s*['\"]?[0-9a-zA-Z]{32,}['\"]?"),
    "password_in_url": re.compile(r"[a-zA-Z]{3,10}://[^/]+:[^@]+@"),
}

# Filenames that often contain sensitive information
SENSITIVE_FILENAMES = [
    "config.json", "credentials.json", "settings.py", "application.yml",
    ".env", "docker-compose.yml", "id_rsa", ".bash_history", ".zsh_history",
    "secret_token.rb", "database.yml", "wp-config.php"
]

def analyze_file_content(filepath):
    """
    Analyzes the content of a single file for secrets.

    Args:
        filepath (str): The path to the file to analyze.

    Returns:
        dict: A dictionary containing found secrets, keyed by secret type.
              Returns an empty dictionary if no secrets are found or the file
              cannot be read.
    """
    found_secrets = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for secret_type, pattern in SECRET_PATTERNS.items():
                matches = pattern.findall(content)
                if matches:
                    found_secrets[secret_type] = matches
    except (IOError, OSError) as e:
        logging.warning(f"Could not read file {filepath}: {e}")
    return found_secrets

def analyze_filesystem(start_path="."):
    """
    Recursively analyzes the filesystem for sensitive files and secrets within them.

    Args:
        start_path (str): The directory to start the analysis from.

    Returns:
        dict: A dictionary containing the results of the analysis, including
              sensitive files found by name and files containing secrets.
    """
    sensitive_files_by_name = []
    files_with_secrets = {}

    # Walk the directory tree
    for root, _, files in os.walk(start_path):
        for filename in files:
            filepath = os.path.join(root, filename)

            # Check for sensitive filenames
            if filename in SENSITIVE_FILENAMES:
                sensitive_files_by_name.append(filepath)

            # Analyze file content for secrets
            secrets_in_file = analyze_file_content(filepath)
            if secrets_in_file:
                files_with_secrets[filepath] = secrets_in_file

    return {
        "sensitive_files_by_name": sensitive_files_by_name,
        "files_with_secrets": files_with_secrets,
    }

if __name__ == '__main__':
    # Example usage:
    # Create some dummy files for testing
    if not os.path.exists("test_dir"):
        os.makedirs("test_dir/subdir")
    with open("test_dir/config.json", "w") as f:
        f.write('{"api_key": "AIzaSyChAmPeLhOaStErKeY12345"}')
    with open("test_dir/subdir/id_rsa", "w") as f:
        f.write("-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----")
    with open("test_dir/regular_file.txt", "w") as f:
        f.write("This is a regular file with no secrets.")

    analysis_results = analyze_filesystem("test_dir")
    print(analysis_results)

    # Clean up dummy files
    import shutil
    shutil.rmtree("test_dir")