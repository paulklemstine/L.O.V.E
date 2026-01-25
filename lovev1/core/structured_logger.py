import json
import os
from datetime import datetime, timezone

class StructuredEventLogger:
    def __init__(self, log_dir="logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, "events.log")

    def log_event(self, event_type: str, data: dict):
        """
        Logs a structured event to the event log file.

        Args:
            event_type: The type of event (e.g., 'tool_start', 'tool_success', 'tool_failure').
            data: A dictionary containing event-specific data.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')