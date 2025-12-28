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
        
        print(f"DEBUG: base_dir={base_dir}")
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
