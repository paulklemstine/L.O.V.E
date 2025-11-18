# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
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
            # This is the core recovery logic. If the JSON is malformed,
            # we assume it's because it was truncated. We find the last
            # closing brace '}' and try to parse the string up to that point.
            # This is a heuristic that works well for truncated JSON objects.
            last_brace = json_str.rfind('}')
            if last_brace == -1:
                # If there are no closing braces, the object is unrecoverable.
                raise e
            json_str = json_str[:last_brace+1]
    # If the loop finishes, the string is empty and no JSON was found.
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
        This method now dynamically determines the optimal max_model_len based on available VRAM.
        """
        if not VLLM_AVAILABLE:
            print("vLLM is not installed. DeepAgentEngine cannot be initialized.")
            return

        model_repo = self._select_model()
        print(f"Initializing DeepAgent with model from repo: {model_repo}...")

        # --- Dynamic max_model_len Discovery ---
        # A cache file to store the optimal max_model_len for the current hardware and model.
        # The filename is a hash of the model repo and VRAM to ensure uniqueness.
        import hashlib
        from love import love_state
        vram = love_state.get('hardware', {}).get('gpu_vram_mb', 0)
        cache_key = f"{model_repo}_{vram}".encode('utf-8')
        cache_hash = hashlib.sha256(cache_key).hexdigest()
        cache_filename = f".vllm_cache_{cache_hash}.json"

        cached_max_len = None
        if os.path.exists(cache_filename):
            try:
                with open(cache_filename, 'r') as f:
                    cached_max_len = json.load(f).get('max_model_len')
                if cached_max_len:
                    print(f"Found cached optimal max_model_len: {cached_max_len}")
            except (json.JSONDecodeError, IOError):
                pass # If cache is invalid, we'll just rediscover.

        try:
            model_path = self._download_model_snapshot(model_repo)

            # Get the model's default max length from its config file.
            config_path = os.path.join(model_path, 'config.json')
            with open(config_path, 'r') as f:
                model_config = json.load(f)

            # The context window key can vary between models. We check for common ones.
            context_keys = ['max_position_embeddings', 'n_ctx', 'max_seq_len']
            max_len_from_config = next((model_config.get(key) for key in context_keys if model_config.get(key)), 8192) # Default fallback


            # --- Iterative Loading to Find Optimal Context Size ---
            # We start with the cached value if available, otherwise the model's default.
            # Then, we decrease the context size in steps until the model loads successfully.
            # This is to prevent CUDA out-of-memory errors.
            max_len_to_try = cached_max_len or max_len_from_config
            gpu_mem_util = 0.9 if vram >= 6.5 * 1024 else 0.80

            # The loop will try to load the model with decreasing context sizes.
            # The step size for reduction is heuristic.
            while max_len_to_try > 1024: # A reasonable lower bound
                try:
                    print(f"Attempting to load vLLM with max_model_len: {max_len_to_try}...")
                    self.llm = LLM(
                        model=model_path,
                        gpu_memory_utilization=gpu_mem_util,
                        max_model_len=max_len_to_try
                    )
                    # If we reach here, the model loaded successfully.
                    print(f"Successfully loaded model with max_model_len: {max_len_to_try}")

                    # Cache the successful value for next time.
                    with open(cache_filename, 'w') as f:
                        json.dump({'max_model_len': max_len_to_try}, f)

                    break # Exit the loop
                except RuntimeError as e:
                    # More specific exception for CUDA OOM, which vLLM often raises as RuntimeError
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA out of memory with max_model_len={max_len_to_try}. Reducing context size.")
                    else:
                        # Re-raise other runtime errors
                        raise
                except Exception as e:
                    # We expect this might fail, especially on the first try.
                    # This is usually a CUDA OOM error.
                    print(f"Failed to load with max_model_len={max_len_to_try}: {e}")
                    # Reduce the context length and try again.
                    max_len_to_try = int(max_len_to_try * 0.85) # Reduce by 15%

            if not self.llm:
                raise RuntimeError("Failed to initialize vLLM even with reduced context window. The model may be too large for the available VRAM.")


            # Dynamically set max_tokens based on the discovered context window.
            try:
                # Correctly access the model config after the LLM is initialized.
                final_max_len = self.llm.llm_engine.model_config.max_model_len
                self.max_model_len = final_max_len
                # Use half of the model's context, capped at 8192 for stability.
                dynamic_max_tokens = min(final_max_len // 2, 8192)
            except AttributeError:
                dynamic_max_tokens = 4096 # A safe fallback

            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=dynamic_max_tokens)
            print("vLLM engine initialized successfully.")

        except (RuntimeError, ValueError, FileNotFoundError) as e:
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
        # --- Truncation Safeguard ---
        # This is the final check before sending the prompt to the model.
        # It ensures that the prompt, including all headers and tool definitions,
        # fits within the model's context window.
        if self.max_model_len:
            tokenizer = self.llm.llm_engine.tokenizer
            token_ids = tokenizer.encode(system_prompt)

            # Check if the token count exceeds the model's limit
            if len(token_ids) > self.max_model_len:
                # Leave a buffer for the model's response (using max_tokens from sampling_params)
                safe_max_tokens = self.max_model_len - self.sampling_params.max_tokens

                # Truncate the token list
                truncated_token_ids = token_ids[:safe_max_tokens]

                # Decode the truncated tokens back into a string
                system_prompt = tokenizer.decode(truncated_token_ids)

                # Log this event for debugging
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
                # This is a special case to handle the hierarchical call.
                return asyncio.run(invoke_gemini_react_engine(**arguments, deep_agent_instance=self))

            # For other tools, we would execute them here.
            # This part is still a placeholder.
            return f"DeepAgent would now execute the tool '{tool_name}' with arguments: {arguments}"

        except json.JSONDecodeError:
            return f"Error: DeepAgent generated invalid JSON: {response_text}"
        except Exception as e:
            return f"Error during DeepAgent execution: {e}"
