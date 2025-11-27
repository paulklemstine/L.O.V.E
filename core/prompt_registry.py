import os
import yaml
from typing import Dict, Any, Optional
from jinja2 import Template
from core.logging import log_event

class PromptRegistry:
    """
    Singleton registry for managing and rendering prompts from a YAML file.
    Supports dynamic reloading and Jinja2 templating.
    """
    _instance = None
    _prompts: Dict[str, str] = {}
    _prompts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.yaml")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptRegistry, cls).__new__(cls)
            cls._instance._load_prompts()
        return cls._instance

    def _load_prompts(self):
        """Loads prompts from the YAML file."""
        if not os.path.exists(self._prompts_file):
            log_event(f"Prompts file not found: {self._prompts_file}", "WARNING")
            return

        try:
            with open(self._prompts_file, 'r') as f:
                self._prompts = yaml.safe_load(f)
            log_event(f"Loaded {len(self._prompts)} prompts from {self._prompts_file}", "INFO")
        except Exception as e:
            log_event(f"Failed to load prompts: {e}", "ERROR")

    def get_prompt(self, key: str) -> Optional[str]:
        """Retrieves a raw prompt template by key."""
        return self._prompts.get(key)

    def render_prompt(self, key: str, **kwargs) -> str:
        """
        Retrieves and renders a prompt template with the provided context.
        
        Args:
            key: The key of the prompt in prompts.yaml
            **kwargs: Variables to inject into the template
            
        Returns:
            Rendered prompt string, or empty string if key not found.
        """
        template_str = self.get_prompt(key)
        if not template_str:
            log_event(f"Prompt key not found: {key}", "ERROR")
            return ""

        try:
            template = Template(template_str)
            return template.render(**kwargs)
        except Exception as e:
            log_event(f"Failed to render prompt '{key}': {e}", "ERROR")
            return template_str  # Return raw template as fallback

    def reload(self):
        """Reloads the prompts from disk."""
        self._load_prompts()

# Global accessor
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
