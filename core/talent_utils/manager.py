import os
import time
import json
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

class ContactManager:
    """
    Manages communication workflows, including rate limiting and encrypted logging.
    """

    def __init__(self, templates, constraints, log_file="contact_log.enc"):
        self.templates = templates
        self.constraints = self._parse_constraints(constraints)
        self.log_file = log_file
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key) if self.encryption_key else None
        self.outreach_history = self._load_outreach_history()

    def _get_encryption_key(self):
        """Retrieves the encryption key from environment variables."""
        key = os.environ.get("TALENT_LOG_KEY")
        if not key:
            print("Warning: TALENT_LOG_KEY not found. Generating a temporary key. Logs will not be persistent across sessions.")
            key = Fernet.generate_key()
        # The key must be 32 url-safe base64-encoded bytes.
        # Let's ensure the provided key is valid, or pad it if necessary
        # For simplicity, we assume the user provides a valid key.
        return key

    def _parse_constraints(self, constraints):
        """Parses string-based time constraints into timedelta objects."""
        parsed = constraints.copy()
        if 'min_response_window' in parsed:
            try:
                # e.g., "7 days" -> timedelta(days=7)
                value, unit = parsed['min_response_window'].split()
                # A simple pluralization check
                if not unit.endswith('s'):
                    unit += 's'
                parsed['min_response_window_td'] = timedelta(**{unit: int(value)})
            except ValueError:
                print(f"Warning: Could not parse time constraint '{parsed['min_response_window']}'. Using default.")
                parsed['min_response_window_td'] = timedelta(days=7)
        return parsed

    def _load_outreach_history(self):
        """Loads and decrypts the outreach history from the log file."""
        history = {}
        if not self.cipher_suite or not os.path.exists(self.log_file):
            return history

        with open(self.log_file, "rb") as f:
            for line in f:
                try:
                    decrypted_data = self.cipher_suite.decrypt(line.strip())
                    log_entry = json.loads(decrypted_data.decode('utf-8'))
                    profile_id = log_entry['anonymized_id']
                    if profile_id not in history:
                        history[profile_id] = []
                    history[profile_id].append(log_entry)
                except Exception as e:
                    print(f"Warning: Could not decrypt or parse a log entry. Skipping. Error: {e}")
        return history

    def _log_attempt(self, profile_id, message_type, message_content):
        """Encrypts and logs a new outreach attempt."""
        if not self.cipher_suite:
            return

        log_entry = {
            'anonymized_id': profile_id,
            'timestamp': datetime.utcnow().isoformat(),
            'message_type': message_type,
            'content': message_content
        }
        encrypted_log = self.cipher_suite.encrypt(json.dumps(log_entry).encode('utf-8'))

        with open(self.log_file, "ab") as f:
            f.write(encrypted_log + b'\n')

        # Update in-memory history
        if profile_id not in self.outreach_history:
            self.outreach_history[profile_id] = []
        self.outreach_history[profile_id].append(log_entry)


    def generate_message(self, message_type, dynamic_slots):
        """Generates a message from a template with dynamic slots."""
        template = self.templates.get(message_type, "")
        if not template:
            return ""

        for key, value in dynamic_slots.items():
            template = template.replace(f"[{key}]", str(value))
        return template

    def can_contact(self, profile_id):
        """Checks if a profile can be contacted based on constraints."""
        history = self.outreach_history.get(profile_id, [])

        # Check max attempts
        if len(history) >= self.constraints.get('max_attempts', 3):
            return False, "Maximum outreach attempts reached."

        # Check response window
        if history:
            last_attempt_time = datetime.fromisoformat(history[-1]['timestamp'])
            min_window = self.constraints.get('min_response_window_td', timedelta(days=7))
            if datetime.utcnow() < last_attempt_time + min_window:
                return False, f"Waiting for response window to elapse. Next contact possible after {last_attempt_time + min_window}."

        return True, "Ready for outreach."

    def record_outreach(self, profile_id, message_type, dynamic_slots):
        """
        Generates a message, logs it, and returns it.
        This is the main public method for this class.
        """
        can_contact_flag, reason = self.can_contact(profile_id)
        if not can_contact_flag:
            print(f"Cannot contact {profile_id}: {reason}")
            return None

        message = self.generate_message(message_type, dynamic_slots)
        if not message:
            print(f"Error: Could not generate message for type '{message_type}'")
            return None

        self._log_attempt(profile_id, message_type, message)
        print(f"Successfully logged outreach to {profile_id}.")

        # In a real system, this would actually send the message.
        # For now, we just return the generated content.
        return message