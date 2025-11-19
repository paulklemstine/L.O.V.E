import os
import json
from datetime import datetime
from cryptography.fernet import Fernet
from core.logging import log_event

class TalentManager:
    """
    Manages the persistent, encrypted database of scouted talent profiles.
    """

    def __init__(self, db_file="talent_database.enc", knowledge_base=None):
        self.db_file = db_file
        self.knowledge_base = knowledge_base
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
        """
        Retrieves the encryption key from environment variables, a config file,
        or generates a new one if not found.
        """
        config_file = "talent_config.json"
        key = os.environ.get("TALENT_LOG_KEY")

        if key:
            log_event("Found TALENT_LOG_KEY in environment variables.", level='INFO')
            return key.encode('utf-8')

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    key = config.get('TALENT_LOG_KEY')
                    if key:
                        log_event(f"Loaded TALENT_LOG_KEY from {config_file}.", level='INFO')
                        return key.encode('utf-8')
            except (json.JSONDecodeError, IOError) as e:
                log_event(f"Error reading {config_file}: {e}. A new key will be generated.", level='WARNING')

        log_event("TALENT_LOG_KEY not found. Generating a new key and saving it to talent_config.json.", level='WARNING')
        try:
            new_key = Fernet.generate_key()
            config_data = {'TALENT_LOG_KEY': new_key.decode('utf-8')}
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            log_event(f"Successfully generated and saved new TALENT_LOG_KEY to {config_file}.", level='INFO')
            return new_key
        except IOError as e:
            log_event(f"CRITICAL: Could not write to {config_file}: {e}. Talent database will be inoperable.", level='CRITICAL')
            return None

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
        Saves a single talent profile to the database and updates the knowledge base graph.
        If a profile with the same anonymized_id already exists, it will be updated.
        """
        """
        Saves a single talent profile to the database and updates the knowledge base graph.
        If a profile with the same anonymized_id already exists, it will be updated.
        """
        if not self.cipher_suite:
            log_event("Cannot save profile: Encryption is not configured.", level='ERROR')
            return "Error: Encryption not configured."

        anonymized_id = profile_data.get('anonymized_id')
        if not anonymized_id:
            log_event("Cannot save profile: Missing 'anonymized_id'.", level='ERROR')
            return "Error: Profile is missing anonymized_id."

        # Initialize relationship fields if they don't exist
        profile_data.setdefault('interaction_history', [])
        profile_data.setdefault('status', 'new')

        # Add a timestamp to track when the profile was last updated
        profile_data['last_saved_at'] = datetime.utcnow().isoformat()
        self.profiles[anonymized_id] = profile_data
        self._save_profiles()  # This is inefficient, but simple and robust for now.

        # --- Knowledge Base Integration ---
        if self.knowledge_base:
            try:
                # Add the talent as a node
                self.knowledge_base.add_node(anonymized_id, 'talent', attributes=profile_data)

                # Extract and add skills as nodes, linking them to the talent
                skills = profile_data.get('skills', [])
                if isinstance(skills, list):
                    for skill in skills:
                        skill_id = f"skill_{skill.lower().replace(' ', '_')}"
                        self.knowledge_base.add_node(skill_id, 'skill', attributes={'name': skill})
                        self.knowledge_base.add_edge(anonymized_id, skill_id, 'HAS_SKILL')
                log_event(f"Updated knowledge base for talent {anonymized_id}.", level='INFO')
            except Exception as e:
                log_event(f"Failed to update knowledge base for talent {anonymized_id}: {e}", level='ERROR')

        log_event(f"Successfully saved profile for {anonymized_id} to the talent database.", level='INFO')
        return f"Successfully saved profile for {profile_data.get('handle', anonymized_id)}."

    def get_profile(self, anonymized_id):
        """Retrieves a single talent profile by their anonymized ID."""
        """Retrieves a single talent profile by their anonymized ID."""
        return self.profiles.get(anonymized_id)

    def add_interaction(self, anonymized_id: str, interaction_type: str, message: str, new_status: str = None):
        """Adds a new interaction to a talent's history and optionally updates their status."""
        """Adds a new interaction to a talent's history and optionally updates their status."""
        if anonymized_id not in self.profiles:
            return "Error: Profile not found."

        profile = self.profiles[anonymized_id]

        # Initialize history if it doesn't exist
        if 'interaction_history' not in profile:
            profile['interaction_history'] = []

        # Add the new interaction
        profile['interaction_history'].append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": interaction_type,
            "message": message
        })

        # Optionally update the status
        if new_status:
            profile['status'] = new_status

        # Save the entire database
        self._save_profiles()
        log_event(f"Added new interaction for {anonymized_id}.", level='INFO')
        return "Interaction added successfully."

    def update_profile_status(self, anonymized_id: str, new_status: str):
        """Updates the status of a specific talent profile."""
        """Updates the status of a specific talent profile."""
        if anonymized_id in self.profiles:
            self.profiles[anonymized_id]['status'] = new_status
            self._save_profiles()
            log_event(f"Updated status for {anonymized_id} to '{new_status}'.", level='INFO')
            return "Status updated successfully."
        return "Error: Profile not found."

    def list_profiles(self):
        """
        Returns a list of summaries for all saved profiles.
        Each summary is a dictionary containing key information.
        """
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
                "status": profile_data.get('status', 'N/A'),
                "last_saved_at": profile_data.get('last_saved_at', 'N/A')
            }
            profile_list.append(summary)
        return profile_list