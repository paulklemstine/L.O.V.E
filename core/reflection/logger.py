import json
import time

class StructuredEventLogger:
    def __init__(self, log_file="event_log.json"):
        self.log_file = log_file

    def log_event(self, event_type, data):
        """Logs a structured event to a file."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        print(f"StructuredEventLogger: Logged event '{event_type}'.")