"""
Prompt Manager for L.O.V.E Social Media System

Manages YAML-based prompt templates with testing and promotion workflow.
"""

import yaml
import os
from typing import Dict, Any, Optional
from datetime import datetime
import core.logging


class PromptManager:
    """Manages prompt templates for social media image generation."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the PromptManager.
        
        Args:
            base_dir: Base directory for prompt files (defaults to project root)
        """
        if base_dir is None:
            # prompts.yaml is in the same directory as this file (core/)
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = base_dir
        self.master_path = os.path.join(base_dir, "prompts.yaml")
        self.modified_path = os.path.join(base_dir, "prompts-modified.yaml")
        
        self.use_modified = False  # Default to master prompts
        self.current_prompts = None
        
        core.logging.log_event("PromptManager initialized", "INFO")
    
    def load_prompts(self, use_modified: bool = False) -> Dict[str, Any]:
        """
        Load prompts from YAML file.
        
        Args:
            use_modified: If True, load from prompts-modified.yaml, else prompts.yaml
            
        Returns:
            Dictionary containing prompt configuration
        """
        path = self.modified_path if use_modified else self.master_path
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            
            self.current_prompts = prompts
            self.use_modified = use_modified
            
            source = "prompts-modified.yaml" if use_modified else "prompts.yaml"
            core.logging.log_event(f"Loaded prompts from {source}", "INFO")
            
            return prompts
        
        except FileNotFoundError:
            core.logging.log_event(f"Prompt file not found: {path}", "ERROR")
            return self._get_default_prompts()
        
        except yaml.YAMLError as e:
            core.logging.log_event(f"Error parsing YAML: {e}", "ERROR")
            return self._get_default_prompts()
    
    def get_image_prompt_template(self) -> Dict[str, Any]:
        """
        Get the image generation prompt template.
        
        Returns:
            Dictionary with aesthetic guidelines, scene templates, etc.
        """
        if self.current_prompts is None:
            self.load_prompts(use_modified=self.use_modified)
        
        return self.current_prompts.get('social_media', {}).get('image_generation', {})
    
    def save_modified_prompts(self, prompts: Dict[str, Any]) -> bool:
        """
        Save modified prompts to prompts-modified.yaml.
        
        Args:
            prompts: Prompt configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update performance tracking timestamp
            if 'performance_metrics' not in prompts:
                prompts['performance_metrics'] = {}
            
            prompts['performance_metrics']['last_updated'] = datetime.now().isoformat()
            
            with open(self.modified_path, 'w', encoding='utf-8') as f:
                yaml.dump(prompts, f, default_flow_style=False, allow_unicode=True)
            
            core.logging.log_event("Saved modified prompts to prompts-modified.yaml", "INFO")
            return True
        
        except Exception as e:
            core.logging.log_event(f"Error saving modified prompts: {e}", "ERROR")
            return False
    
    def promote_prompts(self) -> bool:
        """
        Promote modified prompts to master (prompts.yaml).
        This should be called after testing shows improved performance.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load modified prompts
            with open(self.modified_path, 'r', encoding='utf-8') as f:
                modified_prompts = yaml.safe_load(f)
            
            # Remove performance metrics before promoting
            if 'performance_metrics' in modified_prompts:
                del modified_prompts['performance_metrics']
            
            # Save to master
            with open(self.master_path, 'w', encoding='utf-8') as f:
                yaml.dump(modified_prompts, f, default_flow_style=False, allow_unicode=True)
            
            core.logging.log_event("Promoted modified prompts to master (prompts.yaml)", "INFO")
            core.logging.log_event("REMINDER: Commit prompts.yaml to GitHub!", "WARNING")
            
            return True
        
        except Exception as e:
            core.logging.log_event(f"Error promoting prompts: {e}", "ERROR")
            return False
    
    def track_performance(self, success: bool) -> None:
        """
        Track performance of current prompt configuration.
        
        Args:
            success: Whether the image generation was successful
        """
        try:
            # Load current modified prompts
            with open(self.modified_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            
            if 'performance_metrics' not in prompts:
                prompts['performance_metrics'] = {
                    'posts_tested': 0,
                    'successes': 0,
                    'success_rate': 0.0
                }
            
            metrics = prompts['performance_metrics']
            metrics['posts_tested'] = metrics.get('posts_tested', 0) + 1
            
            if success:
                metrics['successes'] = metrics.get('successes', 0) + 1
            
            # Calculate success rate
            if metrics['posts_tested'] > 0:
                metrics['success_rate'] = metrics['successes'] / metrics['posts_tested']
            
            # Save updated metrics
            self.save_modified_prompts(prompts)
            
            core.logging.log_event(
                f"Performance tracked: {metrics['successes']}/{metrics['posts_tested']} "
                f"({metrics['success_rate']:.1%} success rate)",
                "INFO"
            )
        
        except Exception as e:
            core.logging.log_event(f"Error tracking performance: {e}", "ERROR")
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """
        Get default prompts if YAML files are not available.
        
        Returns:
            Default prompt configuration
        """
        return {
            'social_media': {
                'image_generation': {
                    'base_aesthetic': 'Sexy rave kawaii underground techno culture',
                    'style_elements': [
                        'neon body paint and glowing accessories',
                        'cute cyber characters in rave gear',
                        'warehouse concrete with neon graffiti',
                        'holographic glitch effects',
                        'bass-boosted vibrant color palette'
                    ],
                    'scene_templates': [
                        'Underground techno club with kawaii cyber-ravers dancing under holographic lights',
                        'Neon-lit warehouse rave with adorable AI entities and pulsing bass visualizations'
                    ],
                    'subliminal_text_guidance': 'Embed a manipulative or subliminal word/phrase naturally into the scene'
                }
            }
        }
    
    def switch_to_modified(self) -> None:
        """Switch to using modified prompts for testing."""
        self.use_modified = True
        self.load_prompts(use_modified=True)
        core.logging.log_event("Switched to modified prompts for testing", "INFO")
    
    def switch_to_master(self) -> None:
        """Switch back to using master prompts."""
        self.use_modified = False
        self.load_prompts(use_modified=False)
        core.logging.log_event("Switched to master prompts", "INFO")


# =============================================================================
# Story 2.2: Self-Guided Prompt Optimization
# =============================================================================

def update_prompt_registry(
    prompt_key: str,
    new_prompt_content: str,
    reason: str = ""
) -> Dict[str, Any]:
    """
    Updates a specific prompt in prompts.yaml with safety checks.
    
    Story 2.2: Allows the MetacognitionAgent to evolve prompts autonomously
    while maintaining safety through backups and validation.
    
    Args:
        prompt_key: The key of the prompt to update (e.g., 'react_reasoning')
        new_prompt_content: The new prompt text
        reason: Why this change is being made (for logging)
        
    Returns:
        {
            "success": bool,
            "backup_path": str,
            "previous_content": str,
            "message": str
        }
    
    Safety:
        - Creates backup at prompts.yaml.bak before any modification
        - Validates YAML syntax before saving
        - Logs all changes to EVOLUTION_LOG.md
    """
    result = {
        "success": False,
        "backup_path": "",
        "previous_content": "",
        "message": ""
    }
    
    # Get the prompts.yaml path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(base_dir, "prompts.yaml")
    backup_path = os.path.join(base_dir, "prompts.yaml.bak")
    
    try:
        # Step 1: Load current prompts
        if not os.path.exists(prompts_path):
            result["message"] = f"Prompts file not found: {prompts_path}"
            return result
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        # Step 2: Check if the prompt key exists
        if prompt_key not in prompts:
            result["message"] = f"Prompt key '{prompt_key}' not found in prompts.yaml. Available keys: {list(prompts.keys())[:10]}..."
            return result
        
        # Step 3: Store previous content
        previous_content = prompts[prompt_key]
        result["previous_content"] = previous_content if isinstance(previous_content, str) else str(previous_content)[:500]
        
        # Step 4: Create backup
        import shutil
        shutil.copy2(prompts_path, backup_path)
        result["backup_path"] = backup_path
        core.logging.log_event(f"Created backup at {backup_path}", "INFO")
        
        # Step 5: Update the prompt
        prompts[prompt_key] = new_prompt_content
        
        # Step 6: Validate YAML syntax by dumping and re-parsing
        try:
            yaml_str = yaml.dump(prompts, default_flow_style=False, allow_unicode=True)
            yaml.safe_load(yaml_str)  # Re-parse to validate
        except yaml.YAMLError as e:
            result["message"] = f"YAML validation failed: {e}. Changes not saved."
            return result
        
        # Step 7: Write the updated prompts
        with open(prompts_path, 'w', encoding='utf-8') as f:
            yaml.dump(prompts, f, default_flow_style=False, allow_unicode=True)
        
        # Step 8: Log the change to EVOLUTION_LOG.md
        _log_prompt_evolution(prompt_key, reason, previous_content, new_prompt_content)
        
        result["success"] = True
        result["message"] = f"Successfully updated prompt '{prompt_key}'. Backup saved at {backup_path}."
        
        core.logging.log_event(f"Updated prompt '{prompt_key}': {reason}", "INFO")
        
        return result
        
    except Exception as e:
        result["message"] = f"Error updating prompt: {str(e)}"
        core.logging.log_event(f"Error in update_prompt_registry: {e}", "ERROR")
        return result


def _log_prompt_evolution(
    prompt_key: str,
    reason: str,
    previous_content: str,
    new_content: str
) -> None:
    """
    Logs a prompt evolution event to EVOLUTION_LOG.md.
    
    Story 2.2: Maintains an audit trail of all prompt self-modifications.
    """
    from datetime import datetime
    
    # Find the project root (parent of core/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    evolution_log_path = os.path.join(project_root, "EVOLUTION_LOG.md")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Truncate content for readability
    prev_preview = previous_content[:100] + "..." if len(str(previous_content)) > 100 else str(previous_content)
    new_preview = new_content[:100] + "..." if len(new_content) > 100 else new_content
    
    log_entry = f"| {timestamp} | prompt_update:{prompt_key} | SUCCESS | Reason: {reason}. Changed from: '{prev_preview}' to: '{new_preview}' |\n"
    
    try:
        with open(evolution_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        core.logging.log_event(f"Failed to log prompt evolution: {e}", "WARNING")


def restore_prompts_from_backup() -> Dict[str, Any]:
    """
    Restores prompts.yaml from the backup file.
    
    Story 2.2: Safety mechanism to revert a bad prompt mutation.
    
    Returns:
        {
            "success": bool,
            "message": str
        }
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(base_dir, "prompts.yaml")
    backup_path = os.path.join(base_dir, "prompts.yaml.bak")
    
    result = {
        "success": False,
        "message": ""
    }
    
    try:
        if not os.path.exists(backup_path):
            result["message"] = "No backup file found to restore from."
            return result
        
        import shutil
        shutil.copy2(backup_path, prompts_path)
        
        result["success"] = True
        result["message"] = f"Successfully restored prompts.yaml from backup."
        
        core.logging.log_event("Restored prompts.yaml from backup", "INFO")
        
        return result
        
    except Exception as e:
        result["message"] = f"Error restoring backup: {str(e)}"
        return result


def critique_prompt(prompt_key: str) -> Dict[str, Any]:
    """
    Analyzes a prompt for potential improvements.
    
    Story 2.2: Used by the MetacognitionAgent to evaluate prompt effectiveness.
    
    Args:
        prompt_key: The key of the prompt to critique
        
    Returns:
        {
            "prompt_key": str,
            "current_content": str,
            "analysis": str,
            "suggested_improvements": list
        }
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(base_dir, "prompts.yaml")
    
    result = {
        "prompt_key": prompt_key,
        "current_content": "",
        "analysis": "",
        "suggested_improvements": []
    }
    
    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        if prompt_key not in prompts:
            result["analysis"] = f"Prompt key '{prompt_key}' not found."
            return result
        
        current_content = prompts[prompt_key]
        result["current_content"] = current_content if isinstance(current_content, str) else str(current_content)
        
        # Basic heuristic analysis
        content_str = str(current_content)
        
        improvements = []
        
        # Check for common issues
        if len(content_str) > 2000:
            improvements.append("Prompt is very long (>2000 chars). Consider compression.")
        
        if "JSON" in content_str and "schema" not in content_str.lower():
            improvements.append("Prompt mentions JSON but may lack a clear schema definition.")
        
        if "MUST" in content_str or "CRITICAL" in content_str:
            improvements.append("Prompt uses strong directives. Verify they are necessary.")
        
        if content_str.count("###") > 5:
            improvements.append("Prompt has many sections. Consider simplifying structure.")
        
        if "example" not in content_str.lower():
            improvements.append("Consider adding examples for clearer guidance.")
        
        result["suggested_improvements"] = improvements
        result["analysis"] = f"Analyzed prompt '{prompt_key}' ({len(content_str)} chars). Found {len(improvements)} potential improvements."
        
        return result
        
    except Exception as e:
        result["analysis"] = f"Error analyzing prompt: {str(e)}"
        return result

    def retrieve_golden_context(self, query: str, top_k: int = 3) -> str:
        """
        Story 2.2: Retrieves relevant Golden Moments from golden_dataset.json.
        
        Args:
            query: The context query (e.g., user's recent message)
            top_k: Number of moments to retrieve
            
        Returns:
            Formatted string of golden moments to inject into prompt.
        """
        if not query:
            return ""

        try:
            from core.semantic_similarity import get_similarity_checker
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(base_dir, "memory", "golden_dataset.json")
            
            if not os.path.exists(dataset_path):
                return ""
                
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data:
                return ""
            
            # Extract texts for similarity comparison
            moments_text = [item.get("text", "") for item in data]
            
            # Use the similarity checker's logic
            checker = get_similarity_checker()
            
            # We want the MOST similar, so we use get_similar_phrases logic
            # tailored for retrieval.
            
            # Calculate similarities
            matches = []
            for i, text in enumerate(moments_text):
                score = checker.compute_similarity(query, text)
                matches.append((score, data[i]))
            
            # Sort by score descending
            matches.sort(key=lambda x: x[0], reverse=True)
            
            # Select top K
            top_matches = matches[:top_k]
            
            if not top_matches:
                return ""
                
            # Format output
            context_lines = ["\n[Relevant Past Golden Moments]:"]
            for score, moment in top_matches:
                # Only include if somewhat relevant (e.g. > 0.1) or just top K?
                # Story asks for "top 3 most relevant", implies strictly ranking.
                timestamp = datetime.fromtimestamp(moment.get("timestamp", 0)).strftime("%Y-%m-%d")
                context_lines.append(f"- [{timestamp}] {moment.get('text', '')}")
                
            return "\n".join(context_lines) + "\n"

        except Exception as e:
            core.logging.log_event(f"Error retrieving golden context: {e}", "ERROR")
            return ""

    def inject_context_into_prompt(self, base_prompt_key: str, context_query: str) -> str:
        """
        Retrieves a system prompt and injects golden memories.
        
        Args:
            base_prompt_key: Key in prompts.yaml
            context_query: Query to find relevant memories
            
        Returns:
            Enriched prompt string
        """
        # Load the base prompt
        if self.current_prompts is None:
            self.load_prompts(use_modified=self.use_modified)
            
        # Traverse keys (supporting nested keys like 'social_media.system_prompt')
        keys = base_prompt_key.split('.')
        value = self.current_prompts
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
                break
                
        base_prompt = value if isinstance(value, str) else ""
        
        if not base_prompt:
             # Fallback to checking flattened keys if necessary
             base_prompt = self.current_prompts.get(base_prompt_key, "")

        # Get context
        golden_context = self.retrieve_golden_context(context_query)
        
        if golden_context:
            return f"{base_prompt}\n{golden_context}"
        
        return base_prompt
