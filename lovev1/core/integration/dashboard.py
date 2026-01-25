
import os
import datetime
import logging

class EvolutionDashboard:
    def __init__(self, log_file: str = "EVOLUTION_LOG.md"):
        self.log_file = log_file
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self):
        """Creates the log file with header if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("# Evolution Dashboard Log\n\n")
                f.write("| Timestamp | Function | Status | Details |\n")
                f.write("|---|---|---|---|\n")

    def log_evolution(self, function_name: str, status: str, details: str = ""):
        """
        Appends a log entry to the dashboard log.
        
        Args:
            function_name: The function being evolved.
            status: Status of the evolution (e.g., STARTED, SUCCESS, FAILED).
            details: Additional context or error messages.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Sanitize details for markdown table (replace pipes and newlines)
        clean_details = details.replace("|", "\\|").replace("\n", " ")
        
        entry = f"| {timestamp} | {function_name} | {status} | {clean_details} |\n"
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            logging.error(f"Failed to append to dashboard log: {e}")
