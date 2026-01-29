"""
llm_client.py - Multi-Provider LLM Interface

This module provides a client for LLM generation with multi-provider support.
It automatically detects the best available provider.

Priority:
1. Google Colab AI (free, when running in Colab)
2. Local vLLM server (fastest, no cost)
3. Fallback providers

See docs/llm_client.md for detailed documentation.
"""

import os
import sys
import httpx
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Import Colab detection
from core.colab_llm import is_running_in_colab, ColabLLM, DEFAULT_COLAB_MODEL
from core.llm_parser import strip_thinking_tags

logger = logging.getLogger(__name__)

# Add parent directory to path to import from L.O.V.E. v1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()


class LLMClient:
    """
    Multi-provider LLM client with automatic provider selection.
    
    Priority chain:
    1. Google Colab AI (free, when running in Colab)
    2. Local vLLM server (fastest, no cost)
    3. Fallback providers (if configured)
    """
    
    DEFAULT_VLLM_URL = "http://localhost:8000/v1"
    DEFAULT_TIMEOUT = 120.0
    
    def __init__(
        self, 
        vllm_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        use_fallback: bool = True,
        colab_model: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            vllm_url: URL of the vLLM server. Defaults to localhost:8000.
            timeout: Request timeout in seconds.
            use_fallback: If True, fall back to other providers on failure.
            colab_model: Model to use when in Colab. Defaults to gemini-3-pro-preview.
        """
        self.vllm_url = vllm_url or os.getenv("VLLM_URL", self.DEFAULT_VLLM_URL)
        self.timeout = timeout
        self.use_fallback = use_fallback
        self.model_name: Optional[str] = None
        self._client = httpx.Client(timeout=self.timeout)
        self._async_client: Optional[httpx.AsyncClient] = None
        
        # Colab AI configuration
        self.in_colab = is_running_in_colab()
        self.colab_model = colab_model or DEFAULT_COLAB_MODEL
        self._colab_client: Optional[ColabLLM] = None
        
        if self.in_colab:
            try:
                self._colab_client = ColabLLM(model_name=self.colab_model)
                logger.info(f"🌐 Running in Google Colab - using free Gemini API ({self.colab_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize Colab LLM: {e}")
                self._colab_client = None
        
        
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        # Check if we need to refresh the client due to loop change
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Logic to detect if the existing client is stale (bound to a different/closed loop)
        is_stale = False
        if self._async_client is not None:
             # Check if we stored the creation loop
             creation_loop = getattr(self, '_client_loop', None)
             if creation_loop and creation_loop is not current_loop:
                 is_stale = True
             if self._async_client.is_closed:
                 is_stale = True

        if self._async_client is None or is_stale:
            # If stale, try to close the old one gracefully if possible
            if self._async_client and not self._async_client.is_closed:
                try:
                    await self._async_client.aclose()
                except Exception:
                    pass # Ignore errors closing old client typically

            self._async_client = httpx.AsyncClient(timeout=self.timeout)
            self._client_loop = current_loop
            
        return self._async_client
        
    async def _ensure_model_name_async(self, client: httpx.AsyncClient) -> None:
        """Async version of model name discovery."""
        if self.model_name:
            return
            
        try:
            response = await client.get(f"{self.vllm_url}/models", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    self.model_name = data["data"][0].get("id", "unknown")
        except Exception:
            pass
        
    def _check_vllm_health(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            response = self._client.get(f"{self.vllm_url}/models", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    self.model_name = data["data"][0].get("id", "unknown")
                return True
        except Exception:
            pass
        return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate completion from the LLM.
        
        Args:
            prompt: User prompt/message.
            system_prompt: Optional system message.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            stop: Optional list of stop sequences.
            
        Returns:
            Generated text response.
        """
        # Priority 1: Try Colab AI first (free, no API key needed)
        if self._colab_client is not None:
            try:
                logger.debug(f"Using Colab AI ({self.colab_model}) for generation")
                response = self._colab_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
                return strip_thinking_tags(response)
            except Exception as e:
                logger.warning(f"Colab AI generation failed: {e}, falling back to vLLM")
                if "kernel not ready" in str(e):
                    logger.error("Disabling Colab AI due to permanent kernel error")
                    self._colab_client = None
        
        # Build messages for vLLM
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Priority 2: Try vLLM
        if self._check_vllm_health():
            try:
                response = self._client.post(
                    f"{self.vllm_url}/chat/completions",
                    json={
                        "model": self.model_name or "default",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stop": stop or []
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return strip_thinking_tags(data["choices"][0]["message"]["content"])
            except Exception as e:
                logger.error(f"vLLM error: {e}")
        
        # No providers available
        if self.in_colab:
            raise RuntimeError(f"Colab AI failed and vLLM unavailable at {self.vllm_url}")
        raise RuntimeError(f"vLLM unavailable at {self.vllm_url}")
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> str:
        """Async version of generate()."""
        # Priority 1: Try Colab AI first (note: Colab AI is sync, but we use it anyway)
        if self._colab_client is not None:
            try:
                logger.debug(f"Using Colab AI ({self.colab_model}) for async generation")
                # Colab AI is synchronous, but we can still use it in async context
                response = self._colab_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
                return strip_thinking_tags(response)
            except Exception as e:
                logger.warning(f"Colab AI async generation failed: {e}, falling back to vLLM")
                if "kernel not ready" in str(e):
                    logger.error("Disabling Colab AI due to permanent kernel error")
                    self._colab_client = None
        
        # Build messages for vLLM
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Priority 2: Try vLLM
        client = await self._get_async_client()
        await self._ensure_model_name_async(client)
        
        try:
            response = await client.post(
                f"{self.vllm_url}/chat/completions",
                json={
                    "model": self.model_name or "default",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop or []
                }
            )
            if response.status_code == 200:
                data = response.json()
                return strip_thinking_tags(data["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"Async vLLM error: {e}")
        
        # No providers available
        if self.in_colab:
            raise RuntimeError(f"Colab AI failed and vLLM unavailable at {self.vllm_url}")
        raise RuntimeError(f"vLLM unavailable at {self.vllm_url}")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        Args:
            prompt: Prompt asking for JSON output.
            system_prompt: Optional system message.
            temperature: Lower temperature for more deterministic JSON.
            
        Returns:
            Parsed JSON dictionary.
        """
        json_system = (system_prompt or "") + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation."
        
        # Build messages
        messages = []
        if json_system:
            messages.append({"role": "system", "content": json_system})
        messages.append({"role": "user", "content": prompt})
        
        # Priority 1: Try Colab AI first
        if self._colab_client is not None:
            try:
                # Use standard generation with JSON prompt engineering
                response = self._colab_client.generate(
                    prompt=prompt,
                    system_prompt=json_system
                )
                return self._parse_json(strip_thinking_tags(response))
            except Exception as e:
                logger.warning(f"Colab AI JSON generation failed: {e}, falling back to vLLM")
                if "kernel not ready" in str(e):
                    logger.error("Disabling Colab AI due to permanent kernel error")
                    self._colab_client = None

        # Priority 2: Try vLLM with JSON mode
        if self._check_vllm_health():
            try:
                # Try using response_format (OpenAI compatible)
                response = self._client.post(
                    f"{self.vllm_url}/chat/completions",
                    json={
                        "model": self.model_name or "default",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 4096,  # Increased for JSON
                        "response_format": {"type": "json_object"}
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    return self._parse_json(strip_thinking_tags(text))
            except Exception as e:
                logger.warning(f"vLLM JSON mode error: {e}")
        
        # Fallback to standard generation
        response = self.generate(
            prompt=prompt,
            system_prompt=json_system,
            temperature=temperature,
            max_tokens=4096
        )
        return self._parse_json(response)
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text with fallback strategies."""
        text = text.strip()
        
        # Remove markdown code blocks
        if "```" in text:
            import re
            # Extract content between ```json and ``` or just ``` and ```
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find the first valid JSON object
            import re
            try:
                # Look for outermost matching braces
                # This is a simple heuristic, standard regex can't handle nested braces perfectly
                # but often works for LLM output
                json_match = re.search(r'(\{[\s\S]*\})', text)
                if json_match:
                    return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
            
            # Last ditch: try to fix truncated JSON
            # This is complex, for now just raise with context
            raise ValueError(f"Failed to parse JSON from LLM response: {text[:200]}...")
    
    def close(self):
        """Close HTTP clients."""
        self._client.close()
        if self._async_client:
            # Note: async close should be awaited
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Singleton instance for convenience
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the default LLM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
