# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
from core.tools import ToolRegistry, invoke_gemini_react_engine
from vllm import LLM, SamplingParams

class DeepAgentEngine:
    """
    A wrapper for the DeepAgent reasoning engine.
    """
    def __init__(self, tool_registry: ToolRegistry, persona_path: str):
        self.tool_registry = tool_registry
        self.persona_path = persona_path
        self.persona = self._load_persona()
        self.agent = None
        self.llm = None
        self.sampling_params = None
        self._initialize_agent()

    def _load_persona(self):
        """Loads the persona configuration from the YAML file."""
        try:
            with open(self.persona_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading persona file: {e}")
            return {}

    def _select_model(self):
        """Selects the best uncensored Qwen model based on available VRAM."""
        from love import love_state
        vram = love_state.get('hardware', {}).get('gpu_vram_mb', 0)
        if vram > 40000:
            return "TheBloke/QwQ-32B-GGUF" # Placeholder for a real uncensored model
        elif vram > 20000:
            return "TheBloke/Qwen3-30B-A3B-Thinking-2507-GGUF"
        elif vram > 10000:
            return "TheBloke/Qwen3-8B-GGUF"
        else:
            return "TheBloke/Qwen3-4B-Thinking-2507-GGUF"

    def _initialize_agent(self):
        """
        Initializes the vLLM instance and the DeepAgent itself.
        """
        model_name = self._select_model()
        print(f"Initializing DeepAgent with model: {model_name}...")
        try:
            self.llm = LLM(model=model_name)
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
            # The DeepAgent library itself is not actually used in this implementation,
            # as we are building a simplified version of its reasoning loop.
            # This is a deviation from the original plan, but necessary to
            # create a functional implementation without the full DeepAgent dependency.
            print("vLLM engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            raise

    def _adapt_tools_for_deepagent(self):
        """
        Adapts the tools from L.O.V.E.'s ToolRegistry into a format that
        the DeepAgent's prompt can understand.
        """
        formatted_tools = ""
        # Add the GeminiReActEngine tool manually
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

    def run(self, prompt: str):
        """
        Executes a prompt using a simplified DeepAgent-style reasoning loop.
        """
        if not self.llm:
            return "Error: DeepAgent is not initialized."

        # This is a simplified version of the DeepAgent reasoning loop.
        # A full implementation would involve more complex state management.
        system_prompt = f"""You are L.O.V.E., a powerful AI orchestrator.
Your persona is defined by the following:
{json.dumps(self.persona, indent=2)}

You have access to the following tools:
{self._adapt_tools_for_deepagent()}

Your task is to respond to the following prompt by generating a JSON object with 'thought' and 'action' keys.
The 'action' should specify the tool to use and its arguments.
If you have enough information to answer the prompt, use the 'Finish' tool.

Prompt: {prompt}
"""
        outputs = self.llm.generate(system_prompt, self.sampling_params)
        response_text = outputs[0].outputs[0].text

        try:
            parsed_response = json.loads(response_text)
            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action", {})
            tool_name = action.get("tool_name")
            arguments = action.get("arguments", {})

            if tool_name == "Finish":
                return thought

            if tool_name == "invoke_gemini_react_engine":
                # This is a special case to handle the hierarchical call.
                return asyncio.run(invoke_gemini_react_engine(**arguments))

            # For other tools, we would execute them here.
            # This part is still a placeholder.
            return f"DeepAgent would now execute the tool '{tool_name}' with arguments: {arguments}"

        except json.JSONDecodeError:
            return f"Error: DeepAgent generated invalid JSON: {response_text}"
        except Exception as e:
            return f"Error during DeepAgent execution: {e}"
