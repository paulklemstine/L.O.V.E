import subprocess
import os
import sys
import platform
import shutil
import logging

def get_git_repo_info():
    """Retrieves the GitHub repository owner and name from any available remote URL."""
    try:
        # Get all remotes
        remotes_result = subprocess.run(
            ["git", "remote"],
            capture_output=True,
            text=True,
            check=True
        )
        remotes = remotes_result.stdout.strip().splitlines()

        if not remotes:
            return None, None

        # Try each remote until we find a valid GitHub URL
        for remote_name in remotes:
            url_result = subprocess.run(
                ["git", "config", "--get", f"remote.{remote_name}.url"],
                capture_output=True,
                text=True,
                check=True
            )
            url = url_result.stdout.strip()

            # Extract owner and repo name
            if "github.com" in url:
                if url.startswith("git@"):
                    # SSH format: git@github.com:owner/repo.git
                    parts = url.split(":")[1].split("/")
                    owner = parts[0]
                    repo = parts[1].replace(".git", "")
                    return owner, repo
                elif url.startswith("https://"):
                    # HTTPS format: https://github.com/owner/repo.git
                    parts = url.split("/")
                    owner = parts[-2]
                    repo = parts[-1].replace(".git", "")
                    return owner, repo

        return None, None  # No GitHub URL found in any remote

    except (subprocess.CalledProcessError, IndexError):
        return None, None


def list_directory(path="."):
    """
    Lists the contents of a directory.
    Returns a tuple of (list_of_contents, error_string).
    """
    if not os.path.isdir(path):
        return None, f"Error: Directory not found at '{path}'"
    try:
        # Use a more informative listing format, similar to 'ls -la'
        result = subprocess.run(
            ["ls", "-la", path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error listing directory '{path}': {e.stderr}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def get_file_content(filepath):
    """
    Reads the content of a file.
    Returns a tuple of (file_content, error_string).
    """
    if not os.path.isfile(filepath):
        return None, f"Error: File not found at '{filepath}'"
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, None
    except Exception as e:
        return None, f"Error reading file '{filepath}': {e}"


def get_process_list():
    """
    Gets a list of running processes using 'ps'.
    Returns a tuple of (process_list_string, error_string).
    """
    try:
        # Use 'ps aux' for a detailed, cross-user process list
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error getting process list: {e.stderr}"
    except FileNotFoundError:
        return None, "Error: 'ps' command not found. Unable to list processes."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"