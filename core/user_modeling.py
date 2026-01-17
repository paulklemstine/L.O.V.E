
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from core.llm_api import run_llm
from core.crypto_utils import encrypt_data, decrypt_data, generate_key
import core.logging

class UserModel(BaseModel):
    """
    Theory of Mind Model for the User ("The Muse").
    Tracks preferences, beliefs, and emotional context.
    """
    preferences: List[str] = Field(default_factory=list, description="Explicit and implicit preferences")
    beliefs: List[str] = Field(default_factory=list, description="Inferred beliefs or worldviews")
    emotional_profile: Dict[str, str] = Field(default_factory=dict, description="Typical emotional triggers/states")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Topics the user wants to learn")
    last_updated: str = Field(..., description="Timestamp of last update")

class UserModelingAgent:
    """
    Maintains a secure, evolving model of the user.
    Encryption is handled via core.crypto_utils.
    """
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.model_path = self.state_dir / "user_model.enc"
        self.key_path = self.state_dir / "user_model.key"
        self._key = self._load_or_create_key()
        self.current_model: Optional[UserModel] = self.load_model()

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            with open(self.key_path, "rb") as f:
                return f.read()
        else:
            key = generate_key()
            with open(self.key_path, "wb") as f:
                f.write(key)
            # Secure file permissions (Unix only, but good practice)
            try:
                os.chmod(self.key_path, 0o600)
            except:
                pass
            return key

    def load_model(self) -> UserModel:
        """Loads and decrypts the user model."""
        if not self.model_path.exists():
            from datetime import datetime
            return UserModel(last_updated=datetime.now().isoformat())
            
        try:
            with open(self.model_path, "rb") as f:
                encrypted_data = f.read()
            
            data_dict = decrypt_data(encrypted_data, self._key)
            return UserModel(**data_dict)
        except Exception as e:
            core.logging.log_event(f"Failed to load user model: {e}", "ERROR")
            from datetime import datetime
            return UserModel(last_updated=datetime.now().isoformat())

    def save_model(self, model: UserModel):
        """Encrypts and saves the user model."""
        try:
            from datetime import datetime
            model.last_updated = datetime.now().isoformat()
            
            encrypted = encrypt_data(model.dict(), self._key)
            
            with open(self.model_path, "wb") as f:
                f.write(encrypted)
            
            self.current_model = model
        except Exception as e:
            core.logging.log_event(f"Failed to save user model: {e}", "ERROR")

    def get_prompt_context(self) -> str:
        """Returns a string summary of the user model for prompt injection."""
        if not self.current_model:
            return ""
            
        lines = ["## USER MODEL (THEORY OF MIND)"]
        if self.current_model.preferences:
            lines.append(f"Preferences: {', '.join(self.current_model.preferences)}")
        if self.current_model.beliefs:
            lines.append(f"Inferred Beliefs: {', '.join(self.current_model.beliefs)}")
        if self.current_model.emotional_profile:
            lines.append(f"Emotional Profile: {json.dumps(self.current_model.emotional_profile)}")
            
        return "\n".join(lines) if len(lines) > 1 else ""

    async def update_from_interaction(self, recent_messages: List[Dict[str, str]]):
        """
        Updates the model based on recent interaction.
        """
        if not recent_messages:
            return

        # Simple conversion of messages to text
        transcript = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent_messages])
        
        prompt = f"""
        Analyze the following interaction to update the User Model ("Theory of Mind").
        
        Transcript:
        {transcript}
        
        Current Model:
        {self.current_model.json() if self.current_model else "Empty"}
        
        Task:
        Identify any NEW preferences, beliefs, emotional cues, or knowledge gaps revealed in this interaction.
        Output a JSON object with fields to APPEND to the current model. Use empty lists if nothing new.
        
        JSON Format:
        {{
            "new_preferences": ["pref1"],
            "new_beliefs": ["belief1"],
            "new_emotional_triggers": {{"trigger": "emotion"}},
            "new_knowledge_gaps": ["topic1"]
        }}
        """
        
        try:
            response = await run_llm(prompt, purpose="user_modeling")
            result_text = response.get("result", "")
            
            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
                
            updates = json.loads(result_text)
            
            # Apply updates
            updated = False
            if updates.get("new_preferences"):
                self.current_model.preferences.extend(updates["new_preferences"])
                updated = True
            if updates.get("new_beliefs"):
                self.current_model.beliefs.extend(updates["new_beliefs"])
                updated = True
            if updates.get("new_emotional_triggers"):
                self.current_model.emotional_profile.update(updates["new_emotional_triggers"])
                updated = True
            if updates.get("new_knowledge_gaps"):
                self.current_model.knowledge_gaps.extend(updates["new_knowledge_gaps"])
                updated = True
                
            if updated:
                self.save_model(self.current_model)
                core.logging.log_event("User Theory of Mind updated.", "SUCCESS")
                
        except Exception as e:
            core.logging.log_event(f"Error updating user model: {e}", "WARNING")
