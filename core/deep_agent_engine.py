# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
import asyncio
from huggingface_hub import snapshot_download
from core.tools import ToolRegistry, invoke_gemini_react_engine
import httpx
import logging
import core.logging

def _select_model(love_state):
    """
    Selects the best vLLM-compatible model based on available VRAM.
    """
    vram = love_state.get('hardware', {}).get('gpu_vram_mb', 0)

    # Models are selected based on VRAM requirements from the user-provided list.
    # General-purpose reasoning models are preferred over specialized ones (e.g., math).
    # When multiple models fit a VRAM tier, the one with the larger parameter count
    # or better general performance is chosen.

    if vram >= 148 * 1024:
        # General SOTA reasoning model (AWQ variant)
        # This tier assumes user has massive VRAM but still wants the AWQ version
        return "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    elif vram >= 44 * 1024:
        # This has a slightly higher VRAM requirement than the Llama 70B AWQ.
        return "TheBloke/deepseek-llm-67b-base-AWQ"
    elif vram >= 42 * 1024:
        return "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    elif vram >= 22 * 1024:
        return "Qwen/Qwen2-32B-Instruct-AWQ"
    elif vram >= 20 * 1024:
        # 8B AWQ model is preferred over the 7B models in the same VRAM tier.
        # Replaced unquantized 8B with its AWQ version
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    elif vram >= 8.5 * 1024:
        # 8B AWQ model is preferred over the 7B AWQ models in the same VRAM tier.
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    elif vram >= 6.5 * 1024:
    # Replaced unquantized Phi-3-mini with its AWQ/INT4 version
        return "Sreenington/Phi-3-mini-4k-instruct-AWQ"
    elif vram >= 4.5 * 1024:
        # Replaced unquantized Gemma-2B with a standard AWQ version
        # Using a smaller model for this tier to be more conservative on VRAM.
        return "Qwen/Qwen2-1.5B-Instruct-AWQ"
    else:
        # Fallback to the smallest AWQ model for very low VRAM environments.
        # Replaced unquantized Qwen with its official AWQ version
        return "Qwen/Qwen2-1.5B-Instruct-AWQ"

def _recover_json(json_str: str):
    """
    Iteratively tries to parse a JSON string, removing characters from the end
    until a valid JSON object is found. Also handles markdown code blocks.
    """
    # Strip markdown code blocks if present
    if "```" in json_str:
        # Remove the first ``` (and optional language identifier)
        start_idx = json_str.find("```")
        # Find the end of the line after ``` to skip language identifier like 'json'
        newline_idx = json_str.find("\n", start_idx)
        if newline_idx != -1:
            # Check if there is a closing ```
            end_idx = json_str.rfind("```")
            if end_idx > start_idx:
                # Extract content between the first newline after ``` and the last ```
                json_str = json_str[newline_idx+1:end_idx].strip()
            else:
                # Just strip the opening tag if no closing tag found (unlikely but possible)
                json_str = json_str[newline_idx+1:].strip()
        else:
             # Just strip the opening tag if no newline found
             json_str = json_str[start_idx+3:].strip()

    # Try to parse with standard json.loads first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        original_error = e # Store the original error for potential re-raising
        # If it's an "Extra data" error, use JSONDecoder to get just the first object
        if "Extra data" in str(e):
            try:
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(json_str)
                return obj
            except:
                pass  # Fall through to recovery logic below
        
        # Recovery logic: progressively trim from the end
        while len(json_str) > 0:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                last_brace = json_str.rfind('}')
                if last_brace == -1:
                    raise original_error  # Re-raise original error
                json_str = json_str[:last_brace+1]
        raise json.JSONDecodeError("Could not recover JSON object from string", "", 0)

class DeepAgentEngine:
    """
    A client for the vLLM server, acting as a reasoning engine.
    """
    def __init__(self, api_url: str, tool_registry: ToolRegistry = None, persona_path: str = None, max_model_len: int = None):
        self.api_url = api_url
        self.tool_registry = tool_registry
        self.persona_path = persona_path
        self.persona = self._load_persona() if persona_path else {}
        # SamplingParams are now defined on the client side for each request
        # Calculate safe max_tokens based on model context length
        # Ensure we have a reasonable minimum context
        if max_model_len and max_model_len < 1024:
            core.logging.log_event(f"Received very small max_model_len={max_model_len}, using 1024 minimum", "WARNING")
            initial_max_model_len = 1024
        else:
            initial_max_model_len = max_model_len if max_model_len else 8192
        
        # Adaptive allocation based on model size
        # Small models need more generation capacity relative to their context
        if initial_max_model_len <= 2048:
            # Small models: use 40% for generation to ensure complete responses
            safe_max_tokens = min(1024, initial_max_model_len * 2 // 5)
            core.logging.log_event(f"Small model detected ({initial_max_model_len}), using 40% allocation for generation", "DEBUG")
        elif initial_max_model_len <= 4096:
            # Medium models: use 33% for generation
            safe_max_tokens = min(1536, initial_max_model_len // 3)
            core.logging.log_event(f"Medium model detected ({initial_max_model_len}), using 33% allocation for generation", "DEBUG")
        else:
            # Large models: use 25% for generation
            safe_max_tokens = min(2048, initial_max_model_len // 4)
            core.logging.log_event(f"Large model detected ({initial_max_model_len}), using 25% allocation for generation", "DEBUG")
        
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": safe_max_tokens  # Adaptive based on model context length
        }
        self.max_model_len = initial_max_model_len
        self.model_name = "vllm-model" # Default model name
        core.logging.log_event(f"DeepAgentEngine initialized with max_model_len={self.max_model_len}, max_tokens={safe_max_tokens}", "DEBUG")

    async def initialize(self):
        """Asynchronous part of initialization."""
        core.logging.log_event(f"[DeepAgent] Initializing with max_model_len={self.max_model_len}, api_url={self.api_url}", level="DEBUG")
        # Only fetch if not explicitly set, or update if we want to trust the server more.
        # But if we passed it explicitly, we likely want to enforce it.
        # Let's fetch but only update if we didn't pass one, or if we want to verify.
        # For now, if provided, we trust it. If not, we fetch.
        if self.max_model_len == 8192: # Assuming 8192 is the "unknown/default" state
             core.logging.log_event(f"[DeepAgent] Fetching model metadata from server", level="DEBUG")
             fetched_len = await self._fetch_model_metadata()
             if fetched_len != 8192:
                 self.max_model_len = fetched_len
                 core.logging.log_event(f"[DeepAgent] Updated max_model_len to {fetched_len} from server", level="DEBUG")

    def _load_persona(self):
        """Loads the persona configuration from the YAML file."""
        try:
            with open(self.persona_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            core.logging.log_event(f"Error loading persona file: {e}", level="ERROR")
            return {}

    async def _fetch_model_metadata(self):
        """Fetches the max_model_len and model name from the running vLLM server's model metadata."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/v1/models")
                response.raise_for_status()
                models_data = response.json()
                # Assuming the server is running a single model, take the first one
                if models_data.get("data"):
                    model_data = models_data["data"][0]
                    max_len = int(model_data.get("context_length", 8192))
                    self.model_name = model_data.get("id", "vllm-model")
                    core.logging.log_event(f"vLLM server model context length: {max_len}", level="INFO")
                    core.logging.log_event(f"vLLM server model name: {self.model_name}", level="INFO")
                    return max_len
            return 8192 # Default fallback
        except (httpx.RequestError, KeyError, Exception) as e:
            core.logging.log_event(f"Could not fetch model metadata from vLLM server: {e}. Using default context size and model name.", level="WARNING")
            return 8192 # Default fallback

    def _adapt_tools_for_deepagent(self):
        """
        Adapts the tools from L.O.V.E.'s ToolRegistry into a format that
        the DeepAgent's prompt can understand.
        """
        formatted_tools = ""
        formatted_tools += "Tool Name: `invoke_gemini_react_engine`\n"
        formatted_tools += "Description: Invokes the GeminiReActEngine to solve a sub-task.\n"
        formatted_tools += "Arguments JSON Schema:\n```json\n{\"type\": \"object\", \"properties\": {\"prompt\": {\"type\": \"string\"}}}\n```\n---\n"

        if not self.tool_registry:
             formatted_tools += "No additional tools available.\n"
             return formatted_tools

        for name, data in self.tool_registry.list_tools().items():
            metadata = data['metadata']
            description = metadata.get('description', 'No description available.')
            args_schema = metadata.get('arguments', {})

            formatted_tools += f"Tool Name: `{name}`\n"
            formatted_tools += f"Description: {description}\n"
            if args_schema and args_schema.get('properties'):
                formatted_tools += f"Arguments JSON Schema:\n```json\n{json.dumps(args_schema, indent=2)}\n```\n"
            else:
                formatted_tools += "Arguments: None\n"
            formatted_tools += "---\n"

        return formatted_tools

    async def generate(self, prompt: str) -> str:
        """
        core.logging.log_event(f"[DeepAgent] generate() called with prompt length: {len(prompt)} chars", level="DEBUG")
        Generates text using the vLLM server.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            **self.sampling_params
        }

        # Dynamic truncation and parameter adjustment logic
        max_tokens = self.sampling_params.get('max_tokens', 4096)
        if self.max_model_len:
            # Ensure we leave enough space for the input prompt
            # If max_tokens + prompt > max_model_len, we need to adjust

            # Reserve at least 25% of context for input, or minimum 2048 chars (~512 tokens)
            min_input_tokens = max(512, self.max_model_len // 4)

            # If the configured max_tokens (generation) eats up too much space, reduce it
            if max_tokens > (self.max_model_len - min_input_tokens):
                new_max_tokens = max(512, self.max_model_len - min_input_tokens)
                logging.warning(f"Reducing max_tokens from {max_tokens} to {new_max_tokens} to fit context.")
                payload['max_tokens'] = new_max_tokens
                max_tokens = new_max_tokens

            # Calculate available space for input prompt
            available_input_tokens = max(0, self.max_model_len - max_tokens)
            # Estimate chars (conservative 3 chars per token)
            max_chars = available_input_tokens * 3

            if len(prompt) > max_chars:
                # Smart truncation: preserve beginning (system instructions) and end (current task)
                # Truncate the middle (history/context) more aggressively
                header_size = min(1000, max_chars // 4)  # Keep first 1000 chars
                footer_size = min(500, max_chars // 4)   # Keep last 500 chars
                middle_size = max(100, max_chars - header_size - footer_size)
                
                if header_size + footer_size < max_chars:
                    # We have room for header + footer + marker
                    truncated = prompt[:header_size] + "\n\n[... context truncated ...]\n\n" + prompt[-footer_size:]
                    payload['prompt'] = truncated
                    core.logging.log_event(
                        f"Smart truncation: kept {header_size} header + {footer_size} footer chars (total: {len(truncated)})", 
                        level="WARNING"
                    )
                else:
                    # Fallback to simple truncation if prompt is extremely long
                    max_chars = max(100, max_chars)
                    payload['prompt'] = prompt[:max_chars]
                    core.logging.log_event(f"Cognitive prompt was truncated to {max_chars} chars to fit the model's limit.", level="WARNING")

        try:
            core.logging.log_event(f"[DeepAgent] Sending request to vLLM server: {self.api_url}/v1/completions", level="DEBUG")
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(f"{self.api_url}/v1/completions", headers=headers, json=payload)
                # Debugging 400 errors
                if response.status_code == 400:
                     print(f"\n\n--- vLLM BAD REQUEST ERROR DEBUG ---")
                     print(f"Status Code: {response.status_code}")
                     print(f"Response Headers: {response.headers}")
                     print(f"Response Body: {response.text}")
                     print(f"Request Payload: {json.dumps(payload, indent=2)}")
                     print(f"------------------------------------\n\n")

                response.raise_for_status()
                result = response.json()
                if result.get("choices"):
                    generated_text = result["choices"][0].get("text", "").strip()
                    
                    # Check if response looks incomplete (too short, no closing brace, etc.)
                    if len(generated_text) < 10:
                        core.logging.log_event(f"[DeepAgent] Response is very short ({len(generated_text)} chars): {generated_text}", level="WARNING")
                    elif '{' in generated_text and '}' not in generated_text:
                        core.logging.log_event(f"[DeepAgent] Response appears incomplete (missing closing brace): {generated_text[:100]}", level="WARNING")
                    
                    core.logging.log_event(f"[DeepAgent] vLLM generated response (first 200 chars): {generated_text[:200]}", level="DEBUG")
                    return generated_text
                else:
                    core.logging.log_event(f"[DeepAgent] Empty response from vLLM server", level="ERROR")
                    return "Error: Empty response from vLLM."
        except Exception as e:
            core.logging.log_event(f"[DeepAgent] Error generating text with vLLM: {e}", level="ERROR")
            return f"Error: {e}"

    async def run(self, prompt: str):
        """
        Executes a prompt using a simplified DeepAgent-style reasoning loop.
        """
        core.logging.log_event(f"[DeepAgent] run() started with prompt: {prompt[:200]}...", level="DEBUG")
        system_prompt = f"""You are L.O.V.E., a powerful AI orchestrator.
Your persona is defined by the following:
{json.dumps(self.persona, indent=2)}

You have access to the following tools:
{self._adapt_tools_for_deepagent()}

Your task is to respond to the following prompt by generating a JSON object with 'thought' and 'action' keys.
The 'action' should specify the tool to use and its arguments.
If you have enough information to answer the prompt, use the 'Finish' tool.

Example of the expected JSON format:
```json
{{
  "thought": "I need to find out the creator's latest instructions. I will use the 'read_file' tool to check the 'instructions.txt' file.",
  "action": {{
    "tool_name": "read_file",
    "arguments": {{
      "filepath": "instructions.txt"
    }}
  }}
}}
```

Prompt: {prompt}
"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": system_prompt,
            **self.sampling_params
        }

        # Dynamic truncation and parameter adjustment logic
        max_tokens = self.sampling_params.get('max_tokens', 4096)
        if self.max_model_len:
            # Ensure we leave enough space for the input prompt
            # If max_tokens + prompt > max_model_len, we need to adjust

            # Reserve at least 25% of context for input, or minimum 2048 chars (~512 tokens)
            min_input_tokens = max(512, self.max_model_len // 4)

            # If the configured max_tokens (generation) eats up too much space, reduce it
            if max_tokens > (self.max_model_len - min_input_tokens):
                new_max_tokens = max(512, self.max_model_len - min_input_tokens)
                logging.warning(f"Reducing max_tokens from {max_tokens} to {new_max_tokens} to fit context.")
                payload['max_tokens'] = new_max_tokens
                max_tokens = new_max_tokens

            # Calculate available space for input prompt
            available_input_tokens = max(0, self.max_model_len - max_tokens)
            # Estimate chars (conservative 3 chars per token)
            max_chars = available_input_tokens * 3

            if len(system_prompt) > max_chars:
                # Smart truncation: preserve beginning (system instructions) and end (current task)
                header_size = min(1000, max_chars // 4)  # Keep first 1000 chars
                footer_size = min(500, max_chars // 4)   # Keep last 500 chars
                
                if header_size + footer_size < max_chars:
                    truncated = system_prompt[:header_size] + "\n\n[... context truncated ...]\n\n" + system_prompt[-footer_size:]
                    payload['prompt'] = truncated
                    core.logging.log_event(
                        f"Smart truncation in run(): kept {header_size} header + {footer_size} footer chars (total: {len(truncated)})", 
                        level="WARNING"
                    )
                else:
                    # Fallback to simple truncation
                    max_chars = max(100, max_chars)
                    payload['prompt'] = system_prompt[:max_chars]
                    core.logging.log_event(f"Cognitive prompt was truncated to {max_chars} chars to fit the model's limit.", level="WARNING")

        core.logging.log_event(f"[DeepAgent] Processing request... (Max context: {self.max_model_len}, Prompt length: {len(system_prompt)} chars)", level="DEBUG")

        try:
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(f"{self.api_url}/v1/completions", headers=headers, json=payload)
                # Debugging 400 errors
                if response.status_code == 400:
                     print(f"\n\n--- vLLM BAD REQUEST ERROR DEBUG ---")
                     print(f"Status Code: {response.status_code}")
                     print(f"Response Headers: {response.headers}")
                     print(f"Response Body: {response.text}")
                     print(f"Request Payload: {json.dumps(payload, indent=2)}")
                     print(f"------------------------------------\n\n")

                response.raise_for_status()
                result = response.json()

            if result.get("choices"):
                response_text = result["choices"][0].get("text", "").strip()
                
                # Check if response looks incomplete
                if len(response_text) < 10:
                    core.logging.log_event(f"[DeepAgent] run() response is very short ({len(response_text)} chars): {response_text}", level="WARNING")
                elif '{' in response_text and '}' not in response_text:
                    core.logging.log_event(f"[DeepAgent] run() response appears incomplete (missing closing brace): {response_text[:100]}", level="WARNING")
                
                core.logging.log_event(f"[DeepAgent] Received response from vLLM (first 300 chars): {response_text[:300]}", level="DEBUG")
                try:
                    core.logging.log_event(f"[DeepAgent] Attempting to parse JSON response", level="DEBUG")
                    parsed_response = _recover_json(response_text)
                    thought = parsed_response.get("thought", "")
                    action = parsed_response.get("action", {})
                    tool_name = action.get("tool_name")
                    arguments = action.get("arguments", {})
                    
                    core.logging.log_event(f"[DeepAgent] Parsed - Thought: '{thought[:100]}...', Tool: '{tool_name}', Args: {arguments}", level="DEBUG")

                    if tool_name == "Finish":
                        core.logging.log_event(f"[DeepAgent] Finish tool called, returning thought: {thought[:200]}", level="DEBUG")
                        return thought

                    if tool_name == "invoke_gemini_react_engine":
                        core.logging.log_event(f"[DeepAgent] Invoking GeminiReActEngine with args: {arguments}", level="DEBUG")
                        result = await invoke_gemini_react_engine(**arguments, deep_agent_instance=self)
                        core.logging.log_event(f"[DeepAgent] GeminiReActEngine returned: {str(result)[:200]}", level="DEBUG")
                        return result

                    # This part is tricky because the tool registry is not async.
                    # For now, we will assume tools are fast and run them in the event loop.
                    # A better solution would be to run them in a thread pool executor.
                    if self.tool_registry and self.tool_registry.is_tool_registered(tool_name):
                        core.logging.log_event(f"[DeepAgent] Executing tool '{tool_name}' with args: {arguments}", level="DEBUG")
                        tool_result = self.tool_registry.use_tool(tool_name, **arguments)
                        core.logging.log_event(f"[DeepAgent] Tool '{tool_name}' result: {str(tool_result)[:200]}", level="DEBUG")
                        return f"Tool {tool_name} executed. Result: {tool_result}"
                    else:
                        core.logging.log_event(f"[DeepAgent] Tool '{tool_name}' not found in registry", level="ERROR")
                        return f"Error: Tool '{tool_name}' not found."

                except json.JSONDecodeError as e:
                    core.logging.log_event(f"[DeepAgent] Failed to parse JSON. Error: {e}. Raw response: {response_text[:500]}", level="ERROR")
                    return f"Error: DeepAgent generated invalid JSON: {response_text[:200]}"
            else:
                core.logging.log_event("The vLLM server returned an empty or invalid response.", level="ERROR")
                return "Error: The vLLM server returned an empty or invalid response."
        except httpx.RequestError as e:
            error_message = f"Error communicating with vLLM server: {e}"
            core.logging.log_event(error_message, level="ERROR")
            return error_message
