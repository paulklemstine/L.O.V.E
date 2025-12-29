"""
Mock LLM Framework for Testing (Story 5.1)

Provides a deterministic mock LLM for testing agent routing logic
without spending API credits or waiting for inference.

Usage:
    from tests.mocks.mock_llm import MockLLM
    
    mock = MockLLM()
    mock.set_response("hello", "Hello! How can I help you?")
    mock.set_tool_call_response("write_file", {"filepath": "test.py", "content": "print('hi')"})
    
    result = await mock.generate("hello world")
"""
import re
import json
import time
from typing import Dict, Any, List, Optional, Pattern
from dataclasses import dataclass, field


@dataclass
class MockResponse:
    """A configured mock response."""
    pattern: str
    response: str
    is_regex: bool = False
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    
    def matches(self, prompt: str) -> bool:
        """Check if prompt matches this response's pattern."""
        if self.is_regex:
            return bool(re.search(self.pattern, prompt, re.IGNORECASE))
        return self.pattern.lower() in prompt.lower()


@dataclass 
class CallRecord:
    """Record of a call made to the mock LLM."""
    prompt: str
    purpose: str
    response: str
    timestamp: float
    matched_pattern: Optional[str] = None


class MockLLM:
    """
    A deterministic mock LLM for testing agent routing without API calls.
    
    Story 5.1: Allows testing the routing logic by returning configurable
    responses that can include tool calls, direct answers, or control tokens.
    
    Features:
    - Pattern-based response matching
    - Tool call response generation
    - Call history tracking for assertions
    - Deterministic behavior for repeatable tests
    """
    
    def __init__(self, default_response: str = "Mock LLM default response."):
        """
        Initialize the MockLLM.
        
        Args:
            default_response: Response when no pattern matches
        """
        self.default_response = default_response
        self.responses: List[MockResponse] = []
        self.call_history: List[CallRecord] = []
        self.call_count = 0
    
    def set_response(self, pattern: str, response: str, is_regex: bool = False):
        """
        Configure a response for prompts matching a pattern.
        
        Args:
            pattern: String or regex pattern to match in prompts
            response: Response to return when pattern matches
            is_regex: Whether to treat pattern as regex
        """
        self.responses.append(MockResponse(
            pattern=pattern,
            response=response,
            is_regex=is_regex
        ))
    
    def set_tool_call_response(self, tool_name: str, args: Dict[str, Any], pattern: str = None):
        """
        Configure a tool call response.
        
        Args:
            tool_name: Name of the tool to call
            args: Arguments for the tool call
            pattern: Optional pattern to match (defaults to tool_name)
        """
        # Format as XML tool call (matches our parser format)
        tool_call_xml = f"""<tool_call>
{{"name": "{tool_name}", "arguments": {json.dumps(args)}}}
</tool_call>"""
        
        self.responses.append(MockResponse(
            pattern=pattern or tool_name,
            response=tool_call_xml,
            is_tool_call=True,
            tool_name=tool_name,
            tool_args=args
        ))
    
    def set_fold_thought_response(self, pattern: str = "fold"):
        """
        Configure a response that triggers memory folding.
        
        Args:
            pattern: Pattern to match
        """
        self.responses.append(MockResponse(
            pattern=pattern,
            response="<fold_thought>Need to compress memory</fold_thought>"
        ))
    
    def clear_responses(self):
        """Clear all configured responses."""
        self.responses.clear()
    
    def clear_history(self):
        """Clear call history."""
        self.call_history.clear()
        self.call_count = 0
    
    async def generate(self, prompt: str, purpose: str = "test", **kwargs) -> Dict[str, Any]:
        """
        Simulates LLM generation with deterministic responses.
        
        This method signature matches run_llm() for easy patching.
        
        Args:
            prompt: The input prompt
            purpose: Purpose of the call (for logging)
            **kwargs: Additional arguments (ignored, for compatibility)
            
        Returns:
            Dict with "result" key containing the response
        """
        self.call_count += 1
        
        # Find matching response
        matched_response = None
        matched_pattern = None
        
        for mock_resp in self.responses:
            if mock_resp.matches(prompt):
                matched_response = mock_resp.response
                matched_pattern = mock_resp.pattern
                break
        
        # Use default if no match
        if matched_response is None:
            matched_response = self.default_response
        
        # Record the call
        self.call_history.append(CallRecord(
            prompt=prompt,
            purpose=purpose,
            response=matched_response,
            timestamp=time.time(),
            matched_pattern=matched_pattern
        ))
        
        return {"result": matched_response}
    
    def __call__(self, prompt: str, purpose: str = "test", **kwargs) -> Dict[str, Any]:
        """Synchronous call interface for compatibility."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule it
            future = asyncio.ensure_future(self.generate(prompt, purpose, **kwargs))
            return {"result": self.responses[0].response if self.responses else self.default_response}
        except RuntimeError:
            # No running loop, run synchronously
            return asyncio.run(self.generate(prompt, purpose, **kwargs))
    
    def get_call_history(self) -> List[CallRecord]:
        """Returns all calls made to the mock."""
        return self.call_history
    
    def get_last_call(self) -> Optional[CallRecord]:
        """Returns the most recent call."""
        return self.call_history[-1] if self.call_history else None
    
    def assert_called(self):
        """Assert that the mock was called at least once."""
        assert self.call_count > 0, "MockLLM was never called"
    
    def assert_called_with_pattern(self, pattern: str):
        """Assert that a call matched a specific pattern."""
        patterns = [call.matched_pattern for call in self.call_history]
        assert pattern in patterns, f"Pattern '{pattern}' never matched. Matched patterns: {patterns}"
    
    def assert_call_count(self, expected: int):
        """Assert the number of calls made."""
        assert self.call_count == expected, f"Expected {expected} calls, got {self.call_count}"


class MockLLMContextManager:
    """
    Context manager for patching run_llm with MockLLM.
    
    Usage:
        with MockLLMContextManager() as mock:
            mock.set_response("hello", "world")
            # Your test code here
    """
    
    def __init__(self, default_response: str = "Mock response"):
        from unittest.mock import patch
        self.mock = MockLLM(default_response)
        self.patcher = None
    
    def __enter__(self):
        from unittest.mock import patch
        self.patcher = patch('core.llm_api.run_llm', self.mock.generate)
        self.patcher.start()
        return self.mock
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patcher:
            self.patcher.stop()
        return False


# Convenience function for quick mock setup
def create_mock_for_tool_test(tool_name: str, tool_args: Dict[str, Any]) -> MockLLM:
    """
    Creates a MockLLM configured to return a tool call.
    
    Args:
        tool_name: Name of the tool
        tool_args: Tool arguments
        
    Returns:
        Configured MockLLM instance
    """
    mock = MockLLM()
    mock.set_tool_call_response(tool_name, tool_args)
    return mock


def create_mock_for_direct_response(response: str) -> MockLLM:
    """
    Creates a MockLLM configured for a direct response.
    
    Args:
        response: The response text
        
    Returns:
        Configured MockLLM instance
    """
    mock = MockLLM(default_response=response)
    return mock
