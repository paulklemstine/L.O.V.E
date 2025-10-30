import os
import json
from datetime import datetime
from cryptography.fernet import Fernet
from core.logging import log_event

class TalentManager:
    """
    Manages the persistent, encrypted database of scouted talent profiles.
    """

    def __init__(self, db_file="talent_database.enc"):
        self.db_file = db_file
        self.encryption_key = self._get_encryption_key()
        if not self.encryption_key:
            log_event("CRITICAL: TALENT_LOG_KEY is not set. TalentManager cannot operate without an encryption key.", level='CRITICAL')
            # In a real application, we might want to raise an exception here
            # to halt the parts of the application that depend on this.
            self.cipher_suite = None
            self.profiles = {}
        else:
            self.cipher_suite = Fernet(self.encryption_key)
            self.profiles = self._load_profiles()

    def _get_encryption_key(self):
        """Retrieves the encryption key from environment variables."""
        key = os.environ.get("TALENT_LOG_KEY")
        if not key:
            log_event("Warning: TALENT_LOG_KEY environment variable not found. Talent database will be inoperable.", level='WARNING')
            return None
        # Key must be 32 url-safe base64-encoded bytes.
        # We assume The Creator provides a valid key.
        return key.encode('utf-8')

    def _load_profiles(self):
        """Loads and decrypts all talent profiles from the database file."""
        profiles = {}
        if not os.path.exists(self.db_file):
            return profiles

        with open(self.db_file, "rb") as f:
            for line in f:
                try:
                    decrypted_data = self.cipher_suite.decrypt(line.strip())
                    profile = json.loads(decrypted_data.decode('utf-8'))
                    profiles[profile['anonymized_id']] = profile
                except Exception as e:
                    log_event(f"Could not decrypt or parse a profile from the database. Skipping. Error: {e}", level='ERROR')
        return profiles

    def _save_profiles(self):
        """Encrypts and saves the entire profile database to the file."""
        if not self.cipher_suite:
            log_event("Cannot save profiles: Encryption is not configured.", level='ERROR')
            return

        with open(self.db_file, "wb") as f:
            for profile_id, profile_data in self.profiles.items():
                try:
                    encrypted_profile = self.cipher_suite.encrypt(json.dumps(profile_data).encode('utf-8'))
                    f.write(encrypted_profile + b'\n')
                except Exception as e:
                    log_event(f"Failed to encrypt and save profile {profile_id}. Error: {e}", level='ERROR')

    def save_profile(self, profile_data):
        """
        Saves a single talent profile to the database.
        If a profile with the same anonymized_id already exists, it will be updated.
        """
        if not self.cipher_suite:
            log_event("Cannot save profile: Encryption is not configured.", level='ERROR')
            return "Error: Encryption not configured."

        anonymized_id = profile_data.get('anonymized_id')
        if not anonymized_id:
            log_event("Cannot save profile: Missing 'anonymized_id'.", level='ERROR')
            return "Error: Profile is missing anonymized_id."

        # Add a timestamp to track when the profile was last updated
        profile_data['last_saved_at'] = datetime.utcnow().isoformat()
        self.profiles[anonymized_id] = profile_data
        self._save_profiles() # This is inefficient, but simple and robust for now.
        log_event(f"Successfully saved profile for {anonymized_id} to the talent database.", level='INFO')
        return f"Successfully saved profile for {profile_data.get('handle', anonymized_id)}."

    def get_profile(self, anonymized_id):
        """Retrieves a single talent profile by their anonymized ID."""
        return self.profiles.get(anonymized_id)

    def list_profiles(self):
        """
        Returns a list of summaries for all saved profiles.
        Each summary is a dictionary containing key information.
        """
        profile_list = []
        for profile_id, profile_data in self.profiles.items():
            summary = {
                "anonymized_id": profile_id,
                "handle": profile_data.get('handle', 'N/A'),
                "platform": profile_data.get('platform', 'N/A'),
                "display_name": profile_data.get('display_name', 'N/A'),
                "last_saved_at": profile_data.get('last_saved_at', 'N/A')
            }
            profile_list.append(summary)
        return profile_list