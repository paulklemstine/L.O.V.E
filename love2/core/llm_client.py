"""
llm_client.py - Local vLLM Interface

This module provides a client for interfacing with the local vLLM server.
It prioritizes the local vLLM model over other providers for speed and cost.

See docs/llm_client.md for detailed documentation.
"""

import os
import sys
import httpx
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Add parent directory to path to import from L.O.V.E. v1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()


class LLMClient:
    """
    Client for local vLLM server with fallback to L.O.V.E. v1 llm_api.
    
    Prioritizes:
    1. Local vLLM server (fastest, no cost)
    2. L.O.V.E. v1 llm_api multi-provider pool (fallback)
    """
    
    DEFAULT_VLLM_URL = "http://localhost:8000/v1"
    DEFAULT_TIMEOUT = 120.0
    
    def __init__(
        self, 
        vllm_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        use_fallback: bool = True
    ):
        """
        Initialize the LLM client.
        
        Args:
            vllm_url: URL of the vLLM server. Defaults to localhost:8000.
            timeout: Request timeout in seconds.
            use_fallback: If True, fall back to L.O.V.E. v1 llm_api on vLLM failure.
        """
        self.vllm_url = vllm_url or os.getenv("VLLM_URL", self.DEFAULT_VLLM_URL)
        self.timeout = timeout
        self.use_fallback = use_fallback
        self.model_name: Optional[str] = None
        self._client = httpx.Client(timeout=self.timeout)
        self._async_client: Optional[httpx.AsyncClient] = None
        
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client
        
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
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Try vLLM first
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
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[LLMClient] vLLM error: {e}")
        
        # Fallback to L.O.V.E. v1 llm_api
        if self.use_fallback:
            try:
                from core.llm_api import call_llm
                return call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except ImportError:
                raise RuntimeError("vLLM unavailable and L.O.V.E. v1 llm_api not found")
            except Exception as e:
                raise RuntimeError(f"Both vLLM and fallback failed: {e}")
        
        raise RuntimeError("vLLM unavailable and fallback disabled")
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> str:
        """Async version of generate()."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        client = await self._get_async_client()
        
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
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLMClient] Async vLLM error: {e}")
        
        # Fallback (sync in async context - not ideal but works)
        if self.use_fallback:
            try:
                from core.llm_api import call_llm
                return call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                raise RuntimeError(f"Async generation failed: {e}")
        
        raise RuntimeError("vLLM unavailable and fallback disabled")
    
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
        
        # Try vLLM with JSON mode first
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
                    return self._parse_json(text)
            except Exception as e:
                print(f"[LLMClient] vLLM JSON mode error: {e}")
        
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
