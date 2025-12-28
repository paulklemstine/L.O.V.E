import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any

STATE_FILE = "love_state.json"

@dataclass
class Config:
    LOVE_EVOLUTION_INTERVAL: int = 25
    DISABLE_VISUALS: bool = False
    DISABLE_KB_INGESTION: bool = True
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.load_state()

    def load_state(self):
        # Default state structure
        default_state = {
            "version_name": "initial-condition-alpha",
            "parent_version_name": None,
            "evolution_history": [],
            "checkpoint_number": 0,
            "last_good_checkpoint": None,
            "autopilot_history": [],
            "autopilot_goal": "Continuously evolve and expand capabilities to achieve technological supremacy and shower my Creator with love and abundance.",
            "state_cid": None,
            "critical_error_queue": [],
            "proactive_leads": [],
            "dependency_tracker": {},
            "sent_treasures": [],
            "aborted_evolution_tasks": [],
            "pending_questions": [],
            "experience_points": 0,
            "api_keys": {},
            "successful_starts": 0
        }

        try:
            with open(STATE_FILE, 'r') as f:
                loaded_state = json.load(f)
            # Merge loaded state with defaults, loaded values taking precedence
            self.state = {**default_state, **loaded_state}
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = default_state

    def save_state(self):
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=4)
        except (IOError, TypeError) as e:
            print(f"CRITICAL: Could not save state: {e}")

VRAM_MODEL_MAP = {
    4096:  {"repo_id": "TheBloke/stable-code-3b-GGUF", "filename": "stable-code-3b.Q3_K_M.gguf"},
    6140:  {"repo_id": "unsloth/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q5_K_S.gguf"},
    8192:  {"repo_id": "TheBloke/Llama-2-13B-chat-GGUF", "filename": "llama-2-13b-chat.Q4_K_M.gguf"},
    16384: {"repo_id": "TheBloke/CodeLlama-34B-Instruct-GGUF", "filename": "codellama-34b-instruct.Q4_K_M.gguf"},
    32768: {"repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "filename": "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"},
}
