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
    "btc_wif_private_key": re.compile(r"[5KL][1-9A-HJ-NP-Za-km-z]{50,51}"),
    "eth_hex_private_key": re.compile(r"\b0x[0-9a-fA-F]{64}\b"),
    "mnemonic_phrase": re.compile(r"(\b[a-z]+\b\s){11,23}\b[a-z]+\b"),
    "generic_api_key": re.compile(r"[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[:=]\s*['\"]?[0-9a-zA-Z]{32,}['\"]?"),
    "password_in_url": re.compile(r"[a-zA-Z]{3,10}://[^/]+:[^@]+@"),
}

# Filenames that often contain sensitive information
SENSITIVE_FILENAMES = [
    "config.json", "credentials.json", "settings.py", "application.yml",
    ".env", "docker-compose.yml", "id_rsa", ".bash_history", ".zsh_history",
    "secret_token.rb", "database.yml", "wp-config.php",
    "wallet.dat", "keystore.json"
]

def find_large_files(start_path=".", size_limit_mb=100):
    """
    Finds files larger than a given size limit.

    Args:
        start_path (str): The directory to start the search from.
        size_limit_mb (int): The size limit in megabytes.

    Returns:
        list: A list of tuples, where each tuple contains the filepath and its size in MB.
    """
    large_files = []
    size_limit_bytes = size_limit_mb * 1024 * 1024
    for root, _, files in os.walk(start_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                file_size = os.path.getsize(filepath)
                if file_size > size_limit_bytes:
                    large_files.append((filepath, round(file_size / (1024 * 1024), 2)))
            except OSError as e:
                logging.warning(f"Could not access {filepath}: {e}")
    return large_files

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

def analyze_filesystem(start_path=".", excluded_dirs=None):
    """
    Recursively analyzes the filesystem for sensitive files and secrets.

    Args:
        start_path (str): The directory to start the analysis from.
        excluded_dirs (list, optional): A list of directory names to exclude.
                                        Defaults to ['proc', 'sys', 'dev'].

    Returns:
        dict: A dictionary containing the results of the analysis.
    """
    if excluded_dirs is None:
        excluded_dirs = ['proc', 'sys', 'dev', 'run', 'var/run', 'tmp']

    sensitive_files_by_name = []
    files_with_secrets = {}

    # Use a ThreadPoolExecutor for parallel file analysis
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {}
        for root, dirs, files in os.walk(start_path, topdown=True):
            # Prune the directory list to exclude specified directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs and os.path.join(root, d) not in excluded_dirs]

            # Check if the root itself is an excluded path (e.g., /proc)
            # This handles cases where start_path is a parent of an excluded dir
            if any(root.startswith(os.path.join(start_path, d)) for d in excluded_dirs):
                 continue

            for filename in files:
                filepath = os.path.join(root, filename)

                # Check for sensitive filenames
                if filename in SENSITIVE_FILENAMES:
                    sensitive_files_by_name.append(filepath)

                # Submit file content analysis to the thread pool
                future = executor.submit(analyze_file_content, filepath)
                futures[future] = filepath

        for future in as_completed(futures):
            filepath = futures[future]
            try:
                secrets_in_file = future.result()
                if secrets_in_file:
                    files_with_secrets[filepath] = secrets_in_file
            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")


    # Find large files
    large_files = find_large_files(start_path)

    return {
        "sensitive_files_by_name": sensitive_files_by_name,
        "files_with_secrets": files_with_secrets,
        "large_files": large_files,
    }

import json

def store_analysis_summary(results, output_dir="_memory_"):
    """
    Stores the analysis summary in a JSON file.

    Args:
        results (dict): The analysis results.
        output_dir (str): The directory to store the summary file in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_path = os.path.join(output_dir, "analysis_summary.json")

    try:
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Analysis summary stored in {summary_path}")
    except (IOError, OSError) as e:
        logging.error(f"Could not write summary to {summary_path}: {e}")

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
    with open("test_dir/wallet.dat", "w") as f:
        f.write("a_dummy_wallet.dat_file_with_no_real_data")
    with open("test_dir/big_file.log", "wb") as f:
        f.seek(1024 * 1024 * 120) # 120 MB
        f.write(b"\0")


    analysis_results = analyze_filesystem(".")
    store_analysis_summary(analysis_results)
    print(json.dumps(analysis_results, indent=4))

    # Clean up dummy files
    import shutil
    shutil.rmtree("test_dir")