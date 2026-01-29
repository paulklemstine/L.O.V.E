"""
colab_llm.py - Google Colab AI Integration

This module provides integration with Google Colab's free Gemini AI API.
When running in a Colab environment, this provides free access to Gemini models
without requiring an API key.

Available models (as of 2026):
- google/gemini-3-pro-preview (Pro - complex reasoning, creative tasks)
- google/gemini-2.5-pro (Pro - detailed analysis)
- google/gemini-2.5-flash (Flash - fast responses)
- google/gemini-2.0-flash (Flash - speed-optimized)
- google/gemma-3-27b (Gemma - experimentation)

Usage:
    from core.colab_llm import is_running_in_colab, ColabLLM
    
    if is_running_in_colab():
        llm = ColabLLM()
        response = llm.generate("What is the meaning of life?")
"""

import sys
from typing import Optional, Generator, Any

# Default model - user requested google/gemini-3-pro-preview
DEFAULT_COLAB_MODEL = "google/gemini-3-pro-preview"

# Available Colab models for reference
AVAILABLE_COLAB_MODELS = [
    "google/gemini-3-pro-preview",  # Most capable
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash",
    "google/gemini-2.0-flash-lite",
    "google/gemma-3-27b",
    "google/gemma-3-12b",
    "google/gemma-3-4b",
    "google/gemma-3-1b",
]


def is_running_in_colab() -> bool:
    """
    Detect if we are running in a Google Colab environment.
    
    Returns:
        True if running in Google Colab, False otherwise.
    """
    try:
        # Check if google.colab module is available
        import google.colab
        return True
    except ImportError:
        return False


def get_colab_ai_module():
    """
    Get the google.colab.ai module if available.
    
    Returns:
        The google.colab.ai module or None if not in Colab.
    """
    if not is_running_in_colab():
        return None
    try:
        from google.colab import ai
        return ai
    except ImportError:
        return None


class ColabLLM:
    """
    LLM client for Google Colab's free AI API.
    
    This client wraps google.colab.ai.generate_text() to provide
    a consistent interface with other LLM providers.
    """
    
    def __init__(self, model_name: str = DEFAULT_COLAB_MODEL):
        """
        Initialize the Colab LLM client.
        
        Args:
            model_name: The Colab model to use. Defaults to gemini-3-pro-preview.
        """
        self.model_name = model_name
        self._ai_module = get_colab_ai_module()
        
        if self._ai_module is None:
            raise RuntimeError(
                "ColabLLM can only be used in a Google Colab environment. "
                "The google.colab.ai module is not available."
            )
    
    @property
    def is_available(self) -> bool:
        """Check if Colab AI is available."""
        return self._ai_module is not None
    
    def check_health(self) -> bool:
        """
        Check if the Colab AI environment is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        if self._ai_module is None:
            return False
        # If we could check the kernel status directly, we would here.
        # For now, basic availability is all we can check without making a request.
        return True
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generate text using Google Colab AI.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt (will be prepended to prompt).
            stream: If True, return a generator that yields chunks.
            
        Returns:
            Generated text string, or a generator if stream=True.
        """
        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        if stream:
            return self._generate_stream(full_prompt)
        else:
            return self._generate_batch(full_prompt)
    
    def _generate_batch(self, prompt: str) -> str:
        """Generate text in batch mode (wait for full response)."""
        if self._ai_module is None:
            raise RuntimeError("Colab AI module is not available")
        
        try:
            response = self._ai_module.generate_text(
                prompt,
                model_name=self.model_name
            )
            if response is None:
                raise RuntimeError("Colab AI returned None response")
            return response
        except AttributeError as e:
            # This happens when the kernel isn't fully initialized
            # e.g., 'NoneType' object has no attribute 'kernel'
            if "kernel" in str(e):
                raise RuntimeError(f"Colab AI kernel not ready: {e}") from e
            raise RuntimeError(f"Colab AI generation error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Colab AI generation failed: {e}") from e
    
    def _generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate text in streaming mode (yield chunks as they arrive)."""
        if self._ai_module is None:
            raise RuntimeError("Colab AI module is not available")
        
        try:
            stream = self._ai_module.generate_text(
                prompt,
                model_name=self.model_name,
                stream=True
            )
            if stream is None:
                raise RuntimeError("Colab AI returned None stream")
            for chunk in stream:
                yield chunk
        except AttributeError as e:
            if "kernel" in str(e):
                raise RuntimeError(f"Colab AI kernel not ready: {e}") from e
            raise RuntimeError(f"Colab AI streaming error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Colab AI streaming failed: {e}") from e
    
    def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text with additional context.
        
        Args:
            prompt: The user prompt.
            context: Additional context to include.
            system_prompt: Optional system prompt.
            
        Returns:
            Generated text string.
        """
        full_prompt = f"Context:\n{context}\n\n{prompt}"
        return self.generate(full_prompt, system_prompt=system_prompt)
    
    def list_available_models(self) -> list:
        """
        List available Colab AI models.
        
        Returns:
            List of available model names.
        """
        try:
            # Try to get the list from the API
            return self._ai_module.list_models()
        except (AttributeError, Exception):
            # Fall back to our known list
            return AVAILABLE_COLAB_MODELS.copy()


# Convenience function for quick generation
def colab_generate(
    prompt: str,
    model_name: str = DEFAULT_COLAB_MODEL,
    system_prompt: Optional[str] = None,
    stream: bool = False
) -> str | Generator[str, None, None]:
    """
    Quick generation function for Colab AI.
    
    Args:
        prompt: The prompt to send.
        model_name: Model to use.
        system_prompt: Optional system prompt.
        stream: If True, return a generator.
        
    Returns:
        Generated text or generator.
        
    Raises:
        RuntimeError: If not running in Colab.
    """
    client = ColabLLM(model_name=model_name)
    return client.generate(prompt, system_prompt=system_prompt, stream=stream)
