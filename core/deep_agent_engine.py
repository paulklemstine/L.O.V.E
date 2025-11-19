# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
import asyncio
from huggingface_hub import snapshot_download
from core.tools import ToolRegistry, invoke_gemini_react_engine


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
    until a valid JSON object is found.
    """
    while len(json_str) > 0:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            last_brace = json_str.rfind('}')
            if last_brace == -1:
                raise e
            json_str = json_str[:last_brace+1]
    raise json.JSONDecodeError("Could not recover JSON object from string", "", 0)


import httpx

class DeepAgentEngine:
    """
    A client for the vLLM server, acting as a reasoning engine.
    """
    def __init__(self, api_url: str, tool_registry: ToolRegistry = None, persona_path: str = None):
        self.api_url = api_url
        self.tool_registry = tool_registry
        self.persona_path = persona_path
        self.persona = self._load_persona() if persona_path else {}
        # SamplingParams are now defined on the client side for each request
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096  # Default, can be overridden
        }
        self.max_model_len = 8192 # Initialize with a default

    async def initialize(self):
        """Asynchronous part of initialization."""
        self.max_model_len = await self._fetch_max_model_len()

    def _load_persona(self):
        """Loads the persona configuration from the YAML file."""
        try:
            with open(self.persona_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading persona file: {e}")
            return {}

    async def _fetch_max_model_len(self):
        """Fetches the max_model_len from the running vLLM server's model metadata."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/v1/models")
                response.raise_for_status()
                models_data = response.json()
                # Assuming the server is running a single model, take the first one
                if models_data.get("data"):
                    max_len = models_data["data"][0].get("context_length", 8192)
                    print(f"vLLM server model context length: {max_len}")
                    return max_len
            return 8192 # Default fallback
        except (httpx.RequestError, KeyError) as e:
            print(f"Could not fetch model metadata from vLLM server: {e}. Using default context size.")
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

    async def run(self, prompt: str):
        """
        Executes a prompt using a simplified DeepAgent-style reasoning loop.
        """
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
            "model": "vllm-model",
            "prompt": system_prompt,
            **self.sampling_params
        }

        estimated_tokens = len(system_prompt) // 3
        if self.max_model_len and estimated_tokens > self.max_model_len:
            max_chars = (self.max_model_len - self.sampling_params['max_tokens']) * 3
            payload['prompt'] = system_prompt[:max_chars]
            print(f"WARNING: The cognitive prompt was truncated to fit the model's limit.")

        try:
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(f"{self.api_url}/v1/completions", headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

            if result.get("choices"):
                response_text = result["choices"][0].get("text", "").strip()
                try:
                    parsed_response = _recover_json(response_text)
                    thought = parsed_response.get("thought", "")
                    action = parsed_response.get("action", {})
                    tool_name = action.get("tool_name")
                    arguments = action.get("arguments", {})

                    if tool_name == "Finish":
                        return thought

                    if tool_name == "invoke_gemini_react_engine":
                        return await invoke_gemini_react_engine(**arguments, deep_agent_instance=self)

                    # This part is tricky because the tool registry is not async.
                    # For now, we will assume tools are fast and run them in the event loop.
                    # A better solution would be to run them in a thread pool executor.
                    if self.tool_registry and self.tool_registry.is_tool_registered(tool_name):
                        tool_result = self.tool_registry.use_tool(tool_name, **arguments)
                        return f"Tool {tool_name} executed. Result: {tool_result}"
                    else:
                        return f"Error: Tool '{tool_name}' not found."

                except json.JSONDecodeError:
                    return f"Error: DeepAgent generated invalid JSON: {response_text}"
            else:
                return "Error: The vLLM server returned an empty or invalid response."
        except httpx.RequestError as e:
            error_message = f"Error communicating with vLLM server: {e}"
            print(error_message)
            return error_message
