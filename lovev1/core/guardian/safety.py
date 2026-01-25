
import os
import shutil
import subprocess
import logging
from typing import Optional

class SafetyNet:
    def check_clean_state(self, file_path: str) -> bool:
        """
        Checks if the file has uncommitted changes in git.
        Returns True if clean (no changes), False if dirty or untracked.
        """
        # We need absolute path for git or run from root?
        # git status --porcelain <file>
        try:
            # Assume we run git from the directory of the file or project root?
            # Best to use the directory of the file as cwd for git, or pass full path if git works that way.
            # Using cwd=. usually works if we are in the repo.
            
            # If we are in wsl/linux, paths are straightforward.
            abs_path = os.path.abspath(file_path)
            dirname = os.path.dirname(abs_path)
            filename = os.path.basename(abs_path)
            
            # Run git status on the specific file
            result = subprocess.run(
                ["git", "status", "--porcelain", filename],
                cwd=dirname,
                capture_output=True,
                text=True,
                check=False # We handle return code manually if git fails entirely
            )
            
            if result.returncode != 0:
                logging.error(f"Git check failed for {file_path}: {result.stderr}")
                return False # Treat error as unsafe
                
            if result.stdout.strip():
                logging.warning(f"File {file_path} is dirty: {result.stdout.strip()}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking git status for {file_path}: {e}")
            return False

    def create_backup(self, file_path: str) -> bool:
        """
        Creates a .bak copy of the file.
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"File to backup does not exist: {file_path}")
                return False
                
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            logging.info(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to create backup for {file_path}: {e}")
            return False

    def restore_from_backup(self, file_path: str) -> bool:
        """
        Restores the file from .bak and removes the backup.
        """
        try:
            backup_path = f"{file_path}.bak"
            if not os.path.exists(backup_path):
                logging.error(f"Backup not found for {file_path}")
                return False

            shutil.copy2(backup_path, file_path)
            os.remove(backup_path)
            logging.info(f"Restored {file_path} from backup.")
            return True
        except Exception as e:
            logging.error(f"Failed to restore {file_path}: {e}")
            return False

    def cleanup_backup(self, file_path: str) -> bool:
        """
        Deletes the backup file (called on success).
        """
        try:
            backup_path = f"{file_path}.bak"
            if os.path.exists(backup_path):
                os.remove(backup_path)
                logging.info(f"Removed backup {backup_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to cleanup backup for {file_path}: {e}")
            return False
