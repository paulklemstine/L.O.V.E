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