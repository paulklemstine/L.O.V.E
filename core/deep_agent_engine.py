# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
import asyncio
from huggingface_hub import snapshot_download
from core.tools import ToolRegistry, invoke_gemini_react_engine


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


try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM, SamplingParams = None, None

class DeepAgentEngine:
    """
    A wrapper for the DeepAgent reasoning engine.
    """
    def __init__(self, tool_registry: ToolRegistry, persona_path: str, llm: LLM, sampling_params: SamplingParams):
        self.tool_registry = tool_registry
        self.persona_path = persona_path
        self.persona = self._load_persona()
        self.llm = llm
        self.sampling_params = sampling_params

    def _load_persona(self):
        """Loads the persona configuration from the YAML file."""
        try:
            with open(self.persona_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading persona file: {e}")
            return {}

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

    @property
    def max_model_len(self):
        """
        Returns the maximum model length from the vLLM engine's model config.
        """
        if self.llm and hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'model_config'):
            return self.llm.llm_engine.model_config.max_model_len
        return None

    def run(self, prompt: str):
        """
        Executes a prompt using a simplified DeepAgent-style reasoning loop.
        """
        if not self.llm:
            return "Error: DeepAgent is not initialized."

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
        if self.max_model_len:
            tokenizer = self.llm.llm_engine.tokenizer
            token_ids = tokenizer.encode(system_prompt)

            if len(token_ids) > self.max_model_len:
                safe_max_tokens = self.max_model_len - self.sampling_params.max_tokens
                truncated_token_ids = token_ids[:safe_max_tokens]
                system_prompt = tokenizer.decode(truncated_token_ids)
                print(f"WARNING: The cognitive prompt was truncated to {safe_max_tokens} tokens to fit the model's limit.")

        outputs = self.llm.generate(system_prompt, self.sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        try:
            parsed_response = _recover_json(response_text)
            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action", {})
            tool_name = action.get("tool_name")
            arguments = action.get("arguments", {})

            if tool_name == "Finish":
                return thought

            if tool_name == "invoke_gemini_react_engine":
                return asyncio.run(invoke_gemini_react_engine(**arguments, deep_agent_instance=self))

            return f"DeepAgent would now execute the tool '{tool_name}' with arguments: {arguments}"

        except json.JSONDecodeError:
            return f"Error: DeepAgent generated invalid JSON: {response_text}"
        except Exception as e:
            return f"Error during DeepAgent execution: {e}"
