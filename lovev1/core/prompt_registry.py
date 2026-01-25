import os
import yaml
from typing import Dict, Any, Optional
from jinja2 import Template
from core.logging import log_event

# LangChain Hub imports with graceful fallback
try:
    from langsmith import Client
    from langchain_core.prompts import ChatPromptTemplate
    hub = Client()
except ImportError:
    hub = None
    ChatPromptTemplate = None

class PromptRegistry:
    """
    Singleton registry for managing and rendering prompts from a YAML file.
    Supports dynamic reloading and Jinja2 templating.
    """
    _instance = None
    _prompts: Dict[str, str] = {}
    _remote_cache: Dict[str, str] = {}
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
            with open(self._prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)
            log_event(f"Loaded {len(self._prompts)} prompts from {self._prompts_file}", "INFO")
        except Exception as e:
            log_event(f"Failed to load prompts: {e}", "ERROR")

    
    def get_hub_prompt(self, hub_repo_id: str) -> str:
        """
        Pulls a prompt from LangChain Hub.
        Example: registry.get_hub_prompt("hwchase17/openai-functions-agent")
        """
        if hub is None:
            log_event("LangSmith Client not available. Install 'langsmith'.", "WARNING")
            return self.get_prompt(hub_repo_id.split("/")[-1]) or ""

        try:
            log_event(f"Pulling prompt from Hub: {hub_repo_id}", "INFO")
            prompt = hub.pull_prompt(hub_repo_id)
            
            # Convert to string if you are strictly using text-based prompts
            # or return the PromptTemplate object if your llm_api.py supports it.
            if ChatPromptTemplate and isinstance(prompt, ChatPromptTemplate):
                # For chat prompts, we might want to return the messages or a formatted string.
                # Here we default to the first message's template or a string representation
                if prompt.messages:
                     return prompt.messages[0].prompt.template
                return str(prompt)
            
            if hasattr(prompt, 'template'):
                return prompt.template
                
            return str(prompt)
        except Exception as e:
            log_event(f"Hub Pull Failed for {hub_repo_id}", {"error": str(e)})
            # Fallback to local prompt if possible (using the name part)
            local_key = hub_repo_id.split("/")[-1]
            return self.get_prompt(local_key) or ""

    def push_to_hub(self, repo_id: str, prompt_content: str, **kwargs) -> bool:
        """
        Pushes a prompt to the LangChain Hub.
        Useful for Metacognition agents saving optimized prompts.
        """
        if hub is None:
             log_event("LangSmith Client not available.", "ERROR")
             return False
             
        try:
            from langchain_core.prompts import PromptTemplate
            # Create a simple prompt object to push
            prompt = PromptTemplate.from_template(prompt_content)
            hub.push_prompt(repo_id, object=prompt, **kwargs)
            log_event(f"Successfully pushed prompt to {repo_id}", "INFO")
            return True
        except Exception as e:
            log_event(f"Hub Push Failed for {repo_id}", {"error": str(e)})
            return False

    def get_prompt(self, key: str) -> Optional[str]:
        """
        Retrieves a raw prompt template by key.
        Checks LangChain Hub if enabled.
        """
        # Remote Prompt Logic
        if os.environ.get("USE_REMOTE_PROMPTS", "false").lower() == "true":
            # Check cache first
            if key in self._remote_cache:
                return self._remote_cache[key]
                
            repo_handle = os.environ.get("LANGCHAIN_HUB_REPO", "love-agent")
            full_hub_id = f"{repo_handle}/{key}"
            
            # Use the new explicit method for consistency, but handle return
            hub_content = self.get_hub_prompt(full_hub_id)
            if hub_content:
                 self._remote_cache[key] = hub_content
                 return hub_content
        
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
        self._remote_cache = {} # Clear remote cache to force re-fetch
        log_event("PromptRegistry reloaded from disk/remote settings.", "INFO")

    def update_prompt(self, key: str, value: str) -> bool:
        """Updates a prompt in the YAML file and reloads."""
        try:
            # Create backup
            backup_file = self._prompts_file + ".bak"
            if os.path.exists(self._prompts_file):
                import shutil
                shutil.copy2(self._prompts_file, backup_file)
            
            # Load current data
            with open(self._prompts_file, 'r', encoding='utf-8') as f:
                current_data = yaml.safe_load(f) or {}
            
            current_data[key] = value
            
            # Write back
            with open(self._prompts_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(current_data, f, indent=2, width=4096, allow_unicode=True, default_flow_style=False)
                
            self.reload()
            return True
        except Exception as e:
            log_event(f"Failed to update prompt '{key}': {e}", "ERROR")
            return False

# Global accessor
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
