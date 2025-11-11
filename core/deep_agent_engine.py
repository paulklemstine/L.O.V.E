# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
from huggingface_hub import snapshot_download
from core.tools import ToolRegistry, invoke_gemini_react_engine

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
        """
        Selects the best vLLM-compatible model based on available VRAM.
        """
        from love import love_state
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
        elif vram >= 6.5 * 1024:
            # 8B AWQ model is preferred over the 7B AWQ models in the same VRAM tier.
            return "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
        elif vram >= 4.5 * 1024:
        # Replaced unquantized Phi-3-mini with its AWQ/INT4 version
            return "Sreenington/Phi-3-mini-4k-instruct-AWQ"
        elif vram >= 2.5 * 1024:
            # Replaced unquantized Gemma-2B with a standard AWQ version
            return "TheBloke/Gemma-2B-AWQ"
        else:
            # Fallback to the smallest AWQ model for very low VRAM environments.
            # Replaced unquantized Qwen with its official AWQ version
            return "Qwen/Qwen2-1.5B-Instruct-AWQ"

    def _download_model_snapshot(self, repo_id):
        """
        Downloads a model snapshot from the Hugging Face Hub if it's not already cached
        and returns the local file path.
        """
        try:
            print(f"Ensuring model snapshot for {repo_id} is downloaded...")
            # We can add `allow_patterns` to be more specific if needed, e.g., ["*.safetensors", "*.json"]
            model_path = snapshot_download(repo_id=repo_id)
            print(f"Model snapshot is available at: {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading model snapshot from {repo_id}: {e}")
            raise # Re-raise the exception to be caught by the caller

    def _initialize_agent(self):
        """
        Initializes the vLLM instance and the DeepAgent itself.
        """
        if not VLLM_AVAILABLE:
            print("vLLM is not installed. DeepAgentEngine cannot be initialized.")
            return

        model_repo = self._select_model()
        print(f"Initializing DeepAgent with model from repo: {model_repo}...")
        try:
            # First, ensure the model is downloaded and get the local path.
            model_path = self._download_model_snapshot(model_repo)

            # Now, initialize vLLM with the local, cached path.
            self.llm = LLM(model=model_path, gpu_memory_utilization=0.85)
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
            # The DeepAgent library itself is not actually used in this implementation,
            # as we are building a simplified version of its reasoning loop.
            # This is a deviation from the original plan, but necessary to
            # create a functional implementation without the full DeepAgent dependency.
            print("vLLM engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            self.llm = None # Ensure llm is None on failure
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
                return asyncio.run(invoke_gemini_react_engine(**arguments, deep_agent_instance=self))

            # For other tools, we would execute them here.
            # This part is still a placeholder.
            return f"DeepAgent would now execute the tool '{tool_name}' with arguments: {arguments}"

        except json.JSONDecodeError:
            return f"Error: DeepAgent generated invalid JSON: {response_text}"
        except Exception as e:
            return f"Error during DeepAgent execution: {e}"
