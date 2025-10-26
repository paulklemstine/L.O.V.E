import os
import subprocess
import json
import time
from dataclasses import dataclass
from typing import Dict, Any
from core.logging import log_event, ui_panel_queue
from display import create_retry_panel, get_terminal_width

@dataclass
class GeminiCLIResponse:
    """Data class to hold the response from the gemini-cli."""
    stdout: str
    stderr: str
    return_code: int

class GeminiCLIWrapper:
    """A secure and robust Python wrapper for the gemini-cli executable."""

    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not provided or found in environment variables.")
        self.cli_path = self._find_cli_path()

    def _find_cli_path(self) -> str:
        """Locates the gemini executable within the project's node_modules directory."""
        # This assumes the script is run from the root of the project.
        path = os.path.join("node_modules", ".bin", "gemini")
        if not os.path.exists(path):
            raise FileNotFoundError(f"gemini not found at {path}")
        return path

    def run(self, prompt: str, timeout: int = 60) -> GeminiCLIResponse:
        """
        Invokes the gemini-cli programmatically, passes a prompt via stdin,
        and captures the output. It includes a retry mechanism with exponential backoff for timeouts.
        """
        log_event(f"Invoking gemini-cli with prompt: {prompt}", level="INFO")
        env = os.environ.copy()
        env["GEMINI_API_KEY"] = self.gemini_api_key

        max_retries = 3
        backoff_factor = 2
        initial_backoff = 5 # Start with a 5-second backoff

        for attempt in range(max_retries):
            try:
                process = subprocess.run(
                    [self.cli_path],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                return GeminiCLIResponse(
                    stdout=process.stdout,
                    stderr=process.stderr,
                    return_code=process.returncode,
                )
            except subprocess.TimeoutExpired as e:
                if attempt + 1 >= max_retries:
                    log_event(f"gemini-cli process timed out after {max_retries} attempts.", level="ERROR")
                    raise TimeoutError(f"gemini-cli process timed out after {timeout} seconds.") from e

                backoff_time = initial_backoff * (backoff_factor ** attempt)
                log_event(f"gemini-cli timed out. Retrying in {backoff_time} seconds (Attempt {attempt + 1}/{max_retries})...", level="WARNING")

                # Create and queue the retry panel for the UI
                if ui_panel_queue:
                    width = get_terminal_width()
                    retry_panel = create_retry_panel(
                        model_id="gemini-cli",
                        attempt=attempt + 1,
                        max_attempts=max_retries,
                        backoff_time=backoff_time,
                        purpose="reasoning",
                        width=width
                    )
                    ui_panel_queue.put(retry_panel)

                time.sleep(backoff_time)
            except Exception as e:
                log_event(f"An unexpected error occurred while running gemini-cli: {e}", level="ERROR")
                raise RuntimeError(f"An unexpected error occurred while running gemini-cli: {e}") from e

    def parse_json_output(self, response: GeminiCLIResponse) -> Dict[str, Any]:
        """Parses the JSON output from the gemini-cli."""
        if response.return_code != 0:
            raise ValueError(f"gemini-cli exited with a non-zero exit code: {response.stderr}")
        try:
            return json.loads(response.stdout)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from gemini-cli output: {response.stdout}") from e