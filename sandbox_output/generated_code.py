import os
import logging
import sys
import dotenv
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Any

# Configure secure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CriticalError(Exception):
    """Custom exception for critical errors requiring failsafe activation."""

    pass


class DataProcessor:
    """Class for secure data processing with failsafe mechanisms."""

    def __init__(self, config_path: str = ".env"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .env file securely."""
        try:
            dotenv.load_dotenv(self.config_path)
            return {
                "api_key": os.getenv("API_KEY"),
                "database_url": os.getenv("DATABASE_URL"),
                "timeout": int(os.getenv("REQUEST_TIMEOUT", 30)),
            }
        except dotenv.DotenvException as e:
            logger.critical(f"Configuration error: {e}")
            raise CriticalError("Configuration loading failed") from e

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not self.config["api_key"]:
            logger.critical("API_KEY missing in configuration")
            raise CriticalError("Missing required API key")
        if not self.config["database_url"]:
            logger.critical("DATABASE_URL missing in configuration")
            raise CriticalError("Missing database connection URL")

    def process_data(self, input_file: str, output_file: str) -> None:
        """Process data from input file and save to output file."""
        try:
            # Validate input file
            if not Path(input_file).is_file():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Securely execute external commands
            command = (
                f"python generated_code.py --input {input_file} --output {output_file}"
            )
            with shlex.split(command) as args:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    check=True,  # Ensure non-zero exit codes raise exception
                )

            # Validate execution result
            if result.returncode != 0:
                logger.error(f"Command execution failed: {result.stderr}")
                raise CriticalError("External command failed")

            logger.info(f"Data processed successfully to {output_file}")

        except (FileNotFoundError, PermissionError) as e:
            logger.critical(f"File access error: {e}")
            raise CriticalError("File access failure") from e
        except subprocess.CalledProcessError as e:
            logger.critical(f"Command execution error: {e}")
            raise CriticalError("External command failed") from e
        except Exception as e:
            logger.exception(f"Unexpected processing error: {e}")
            raise CriticalError("Unrecoverable processing error") from e


def main():
    """Main execution function with failsafe handling."""
    try:
        processor = DataProcessor()
        processor.process_data("input_data.csv", "processed_data.json")
    except CriticalError as e:
        logger.critical(f"Failsafe triggered: {e}")
        # Implement failsafe actions here (e.g., shutdown, rollback)
        sys.exit(1)


if __name__ == "__main__":
    main()
