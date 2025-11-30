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
    elif vram >= 4.5 * 1024:
    # Replaced unquantized Phi-3-mini with its AWQ/INT4 version
        return "Sreenington/Phi-3-mini-4k-instruct-AWQ"
    elif vram >= 2.5 * 1024:
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
                # If the last brace is at the very end, we need to find the *previous* one
                # to ensure we are actually shrinking the string.
                if last_brace == len(json_str) - 1:
                    last_brace = json_str.rfind('}', 0, last_brace)
                
                if last_brace == -1:
                    raise original_error  # Re-raise original error if no more objects found
                
                json_str = json_str[:last_brace+1]
        raise json.JSONDecodeError("Could not recover JSON object from string", "", 0)

class DeepAgentEngine:
    """
    A client for the vLLM server, acting as a reasoning engine.
    """
    def __init__(self, api_url: str, tool_registry: ToolRegistry = None, persona_path: str = None, 
                 max_model_len: int = None, knowledge_base=None, memory_manager=None, use_pool: bool = False):
        self.api_url = api_url
        self.tool_registry = tool_registry
        self.persona_path = persona_path
        self.persona = self._load_persona() if persona_path else {}
        self.knowledge_base = knowledge_base
        self.memory_manager = memory_manager
        self.use_pool = use_pool
        
        # SamplingParams are now defined on the client side for each request
        # Calculate safe max_tokens based on model context length
        # Ensure we have a reasonable minimum context
        # Ensure a sensible default if max_model_len is None, zero, or negative
        if not max_model_len or max_model_len <= 0:
            core.logging.log_event("max_model_len missing or non-positive; defaulting to 8192", "WARNING")
            initial_max_model_len = 8192
        elif max_model_len < 1024:
            core.logging.log_event(f"Received very small max_model_len={max_model_len}, using 1024 minimum", "WARNING")
            initial_max_model_len = 1024
        else:
            initial_max_model_len = max_model_len
        
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
        core.logging.log_event(f"DeepAgentEngine initialized with max_model_len={self.max_model_len}, max_tokens={safe_max_tokens}, use_pool={self.use_pool}", "DEBUG")

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
                    # Guard against zero or negative context lengths returned by the server
                    if max_len <= 0:
                        core.logging.log_event("vLLM reported zero/negative context length; defaulting to 8192", "WARNING")
                        max_len = 8192
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

    def _get_kb_context(self, prompt: str, max_tokens: int = 400) -> str:
        """
        Retrieves relevant context from knowledge base and memories.
        
        Args:
            prompt: The current prompt/goal
            max_tokens: Maximum tokens to use for context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Get KB summary if available
        if self.knowledge_base:
            try:
                from core.kb_tools import get_kb_summary
                kb_summary = get_kb_summary(self.knowledge_base, max_tokens=max_tokens // 2)
                if kb_summary and "not available" not in kb_summary.lower():
                    context_parts.append("📚 Knowledge Base Context:")
                    context_parts.append(kb_summary)
            except Exception as e:
                core.logging.log_event(f"[DeepAgent] Failed to get KB summary: {e}", level="DEBUG")
        
        # Get relevant memories if available
        if self.memory_manager:
            try:
                from core.kb_tools import search_memories
                memories_json = search_memories(prompt, top_k=2, memory_manager=self.memory_manager)
                memories_data = json.loads(memories_json)
                if memories_data.get("count", 0) > 0:
                    context_parts.append("\n🧠 Relevant Past Experiences:")
                    for memory in memories_data.get("memories", []):
                        context_parts.append(f"  - {memory}")
            except Exception as e:
                core.logging.log_event(f"[DeepAgent] Failed to get memories: {e}", level="DEBUG")
        
        if context_parts:
            return "\n".join(context_parts) + "\n"
        return ""


    async def generate(self, prompt: str) -> str:
        """
        core.logging.log_event(f"[DeepAgent] generate() called with prompt length: {len(prompt)} chars", level="DEBUG")
        Generates text using the vLLM server.
        """
        # Apply prompt compression if applicable
        from core.prompt_compressor import compress_prompt, should_compress
        
        original_prompt = prompt
        if should_compress(prompt, purpose="deep_agent_generation"):
            compression_result = compress_prompt(
                prompt,
                purpose="deep_agent_generation"
            )
            if compression_result["success"]:
                prompt = compression_result["compressed_text"]
                core.logging.log_event(
                    f"[DeepAgent] generate() compressed: {compression_result['original_tokens']} → "
                    f"{compression_result['compressed_tokens']} tokens ({compression_result['ratio']:.1%})",
                    level="DEBUG"
                )
        
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

    async def _repair_json_with_llm(self, malformed_text: str, error_context: str) -> dict:
        """
        Uses the LLM to attempt to repair a malformed JSON response.
        """
        core.logging.log_event(f"[DeepAgent] Attempting to repair JSON with LLM...", level="WARNING")
        
        from core.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        repair_prompt = registry.render_prompt(
            "deep_agent_json_repair",
            error_context=error_context,
            malformed_text=malformed_text
        )
        try:
            if self.use_pool:
                from core.llm_api import run_llm
                # Use a fast model for repair if possible, but for now just use the pool default
                result_dict = await run_llm(repair_prompt, purpose="json_repair", deep_agent_instance=None)
                repaired_text = result_dict.get("result", "").strip()
            else:
                # Use the existing generate method for vLLM
                # We might want to use a lower temperature for repair
                original_temp = self.sampling_params.get("temperature")
                self.sampling_params["temperature"] = 0.1 # Low temp for deterministic repair
                repaired_text = await self.generate(repair_prompt)
                self.sampling_params["temperature"] = original_temp # Restore temp

            # Try to parse the repaired text
            return _recover_json(repaired_text)
        except Exception as e:
            core.logging.log_event(f"[DeepAgent] JSON repair failed: {e}", level="ERROR")
            return None



    async def _validate_and_execute_tool(self, parsed_response: dict) -> dict:
        """
        Validates the parsed JSON response and executes the requested tool.
        """
        # Validate response structure
        if not isinstance(parsed_response, dict):
            core.logging.log_event(f"[DeepAgent] Invalid response: expected dict, got {type(parsed_response)}", level="ERROR")
            return {"result": f"Error: LLM returned invalid response type: {type(parsed_response)}", "thought": "Invalid response type"}
        
        # Check for required keys
        if "thought" not in parsed_response or "action" not in parsed_response:
            # Fallback for 'command'/'arguments' format (common in some finetunes)
            if "command" in parsed_response:
                core.logging.log_event("[DeepAgent] Detected 'command'/'arguments' format. converting to thought/action.", level="WARNING")
                cmd = parsed_response.get("command")
                args = parsed_response.get("arguments", {})
                
                tool_name = cmd
                
                # Attempt to handle "tool_name argument" format if the full cmd is not a known tool
                if self.tool_registry and tool_name not in self.tool_registry.list_tools():
                    parts = cmd.split(maxsplit=1)
                    if len(parts) == 2:
                        potential_tool = parts[0]
                        potential_arg = parts[1]
                        
                        if potential_tool in self.tool_registry.list_tools():
                            tool_name = potential_tool
                            core.logging.log_event(f"[DeepAgent] Inferred tool '{tool_name}' and arg '{potential_arg}' from command string.", level="DEBUG")
                            
                            # If no args provided, try to map the string to the first argument
                            if not args:
                                tool_data = self.tool_registry.list_tools().get(tool_name)
                                if tool_data:
                                    props = tool_data['metadata'].get('arguments', {}).get('properties', {})
                                    if props:
                                        # Take the first property key
                                        first_arg = list(props.keys())[0]
                                        args[first_arg] = potential_arg
                                        core.logging.log_event(f"[DeepAgent] Auto-mapped argument '{potential_arg}' to '{first_arg}'", level="DEBUG")

                parsed_response["thought"] = f"Decided to execute command: {cmd}"
                parsed_response["action"] = {"tool_name": tool_name, "arguments": args}
            else:
                # Attempt LLM repair as a final fallback
                core.logging.log_event(f"[DeepAgent] Invalid keys found: {list(parsed_response.keys())}. Attempting LLM repair...", level="WARNING")
                # We need to be careful not to infinite loop here if repair returns invalid keys again.
                # Since we are already in a helper, let's assume the caller handles major repairs, 
                # but we can try one more specific repair for keys.
                repaired_response = await self._repair_json_with_llm(json.dumps(parsed_response), f"Missing keys. Got: {list(parsed_response.keys())}")
                
                if repaired_response and isinstance(repaired_response, dict) and "thought" in repaired_response and "action" in repaired_response:
                    core.logging.log_event("[DeepAgent] JSON successfully repaired by LLM.", level="INFO")
                    parsed_response = repaired_response
                else:
                    core.logging.log_event(
                        f"[DeepAgent] Invalid response structure. Expected {{\"thought\": \"...\", \"action\": {{...}}}}. "
                        f"Got keys: {list(parsed_response.keys())}. Response: {parsed_response}",
                        level="ERROR"
                    )
                    return {"result": f"Error: LLM returned wrong format. Expected 'thought' and 'action' keys, got: {list(parsed_response.keys())}", "thought": "Invalid response structure"}
        
        thought = parsed_response.get("thought", "")
        action = parsed_response.get("action", {})
        
        if not isinstance(action, dict):
            # Handle case where action is a string (common with some models)
            if isinstance(action, str):
                core.logging.log_event(f"[DeepAgent] Action is a string: '{action}'. Attempting to parse or wrap.", level="WARNING")
                
                # Case 1: The action is just the tool name (e.g. "Finish")
                if action.strip() == "Finish":
                    core.logging.log_event("[DeepAgent] Action string is 'Finish'. Wrapping in dict.", level="DEBUG")
                    action = {"tool_name": "Finish", "arguments": {}}
                
                # Case 2: The action string is actually a JSON string
                elif action.strip().startswith("{"):
                    try:
                        parsed_action = _recover_json(action)
                        if isinstance(parsed_action, dict):
                            action = parsed_action
                            core.logging.log_event(f"[DeepAgent] Successfully parsed action string as JSON: {action}", level="DEBUG")
                        else:
                            core.logging.log_event(f"[DeepAgent] Action string parsed but not a dict: {type(parsed_action)}", level="WARNING")
                    except Exception as e:
                        core.logging.log_event(f"[DeepAgent] Failed to parse action string as JSON: {e}", level="WARNING")
                
                # Case 3: It's likely a tool name but we don't have args.
                else:
                    # Check if it matches a known tool
                    known_tools = list(self.tool_registry.list_tools().keys()) if self.tool_registry else []
                    if action.strip() in known_tools:
                            core.logging.log_event(f"[DeepAgent] Action string '{action}' matches a known tool. Wrapping.", level="DEBUG")
                            action = {"tool_name": action.strip(), "arguments": {}}
                    else:
                        # If we can't figure it out, return a helpful error
                        core.logging.log_event(f"[DeepAgent] Could not convert string action '{action}' to dict.", level="ERROR")
                        return {"result": f"Error: 'action' field was a string ('{action}'), but expected a dictionary like {{'tool_name': '...', 'arguments': {{...}}}}. Please use the correct format.", "thought": thought}

            if not isinstance(action, dict):
                core.logging.log_event(f"[DeepAgent] Invalid action: expected dict, got {type(action)}", level="ERROR")
                return {"result": f"Error: 'action' must be a dict, got {type(action)}", "thought": thought}
        
        tool_name = action.get("tool_name")
        arguments = action.get("arguments", {})
        
        if not tool_name:
            core.logging.log_event(
                f"[DeepAgent] Missing tool_name in action. Action: {action}", 
                level="ERROR"
            )
            return {"result": f"Error: 'tool_name' is required in action. Got action: {action}", "thought": thought}
        
        core.logging.log_event(f"[DeepAgent] Parsed - Thought: '{thought[:100]}...', Tool: '{tool_name}', Args: {arguments}", level="DEBUG")

        if tool_name == "Finish":
            core.logging.log_event(f"[DeepAgent] Finish tool called, returning thought: {thought[:200]}", level="DEBUG")
            return {"result": thought, "thought": thought}

        if tool_name == "invoke_gemini_react_engine":
            if "prompt" not in arguments:
                error_msg = "Error: 'prompt' argument is required for invoke_gemini_react_engine. Please provide the goal or question for the sub-agent."
                core.logging.log_event(f"[DeepAgent] {error_msg}", level="ERROR")
                return {"result": error_msg, "thought": thought}

            core.logging.log_event(f"[DeepAgent] Invoking GeminiReActEngine with args: {arguments}", level="DEBUG")
            result = await invoke_gemini_react_engine(**arguments, deep_agent_instance=self)
            core.logging.log_event(f"[DeepAgent] GeminiReActEngine returned: {str(result)[:200]}", level="DEBUG")
            return {"result": result, "thought": thought}

        # Execute tool from registry
        if self.tool_registry and tool_name in self.tool_registry.list_tools():
            core.logging.log_event(f"[DeepAgent] Executing tool '{tool_name}' with args: {arguments}", level="DEBUG")
            try:
                tool_func = self.tool_registry.get_tool(tool_name)
                if asyncio.iscoroutinefunction(tool_func):
                    tool_result = await tool_func(**arguments)
                else:
                    tool_result = tool_func(**arguments)

                core.logging.log_event(f"[DeepAgent] Tool '{tool_name}' result: {str(tool_result)[:200]}", level="DEBUG")
                return {"result": f"Tool {tool_name} executed. Result: {tool_result}", "thought": thought}
            except Exception as e:
                    core.logging.log_event(f"[DeepAgent] Tool '{tool_name}' execution failed: {e}", level="ERROR")
                    return {"result": f"Error executing tool '{tool_name}': {e}", "thought": thought}
        else:
            available_tools_list = list(self.tool_registry.list_tools().keys()) if self.tool_registry else []
            error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(available_tools_list[:10])}"
            
            # Provide specific guidance for common mistakes
            if tool_name in ["Knowledge Base", "Memory", "knowledge_base", "memory"]:
                error_msg += "\n\nNOTE: 'Knowledge Base' and 'Memory' are NOT tools. They are informational context provided in the prompt. Please use one of the actual tools listed above."
            elif tool_name in ["JSON Repair Expert", "json_repair", "repair_json"]:
                error_msg += "\n\nNOTE: 'JSON Repair Expert' is NOT a tool. If you need to fix malformed output, simply use the 'Finish' tool to return your corrected response. Do not try to call non-existent repair tools."
            
            core.logging.log_event(f"[DeepAgent] {error_msg}", level="ERROR")
            return {"result": error_msg, "thought": thought}
    async def run(self, prompt: str):
        """
        Executes a prompt using a simplified DeepAgent-style reasoning loop.
        """
        core.logging.log_event(f"[DeepAgent] run() started with prompt: {prompt[:200]}...", level="DEBUG")
        
        # Track recent actions for loop detection
        recent_actions = []
        iteration_count = 0
        
        # Get Knowledge Base Context
        kb_context = self._get_kb_context(prompt)
        
        from core.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        
        system_prompt = registry.render_prompt(
            "deep_agent_system",
            persona_json=json.dumps(self.persona, indent=2),
            tools_desc=self._adapt_tools_for_deepagent(),
            kb_context=kb_context,
            prompt=prompt
        )
        core.logging.log_event(f"[DeepAgent] Processing request... (Max context: {self.max_model_len}, Prompt length: {len(system_prompt)} chars)", level="DEBUG")

        # Apply prompt compression if applicable
        from core.prompt_compressor import compress_prompt, should_compress
        
        if should_compress(system_prompt, purpose="deep_agent_reasoning"):
            # Collect force tokens: tool names and critical keywords
            force_tokens = ["tool_name", "arguments", "thought", "action", "Finish", "JSON"]
            if self.tool_registry:
                force_tokens.extend(list(self.tool_registry.list_tools().keys()))
            
            compression_result = compress_prompt(
                system_prompt,
                force_tokens=force_tokens,
                purpose="deep_agent_reasoning"
            )
            
            if compression_result["success"]:
                system_prompt = compression_result["compressed_text"]
                core.logging.log_event(
                    f"[DeepAgent] Compressed prompt: {compression_result['original_tokens']} → "
                    f"{compression_result['compressed_tokens']} tokens "
                    f"({compression_result['ratio']:.1%} compression) in {compression_result['time_ms']:.0f}ms",
                    level="DEBUG"
                )

        try:
            if self.use_pool:
                from core.llm_api import run_llm
                core.logging.log_event("[DeepAgent] Using LLM Pool for generation.", level="DEBUG")
                # We pass deep_agent_instance=None to prevent run_llm from trying to use us (recursion)
                result_dict = await run_llm(system_prompt, purpose="deep_agent_reasoning", deep_agent_instance=None)
                response_text = result_dict.get("result", "").strip()
                
                if not response_text:
                    core.logging.log_event("[DeepAgent] Empty response from LLM Pool.", level="ERROR")
                    return "Error: Empty response from LLM Pool."
                    
            else:
                # Existing vLLM Logic
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
                else:
                    response_text = ""

            # Common response processing
            if response_text:
                # Check if response looks incomplete
                if len(response_text) < 10:
                    core.logging.log_event(f"[DeepAgent] run() response is very short ({len(response_text)} chars): {response_text}", level="WARNING")
                elif '{' in response_text and '}' not in response_text:
                    core.logging.log_event(f"[DeepAgent] run() response appears incomplete (missing closing brace): {response_text[:100]}", level="WARNING")
                
                core.logging.log_event(f"[DeepAgent] Received response (first 300 chars): {response_text[:300]}", level="DEBUG")
                
                # Try to extract JSON from the response
                # The model sometimes outputs conversational text before/after the JSON
                json_text = response_text
                
                # Look for JSON object in the response
                if '{' in response_text and '}' in response_text:
                    # Find the first { and last }
                    first_brace = response_text.find('{')
                    last_brace = response_text.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_text = response_text[first_brace:last_brace+1]
                        if json_text != response_text:
                            core.logging.log_event(f"[DeepAgent] Extracted JSON from conversational text. Original length: {len(response_text)}, JSON length: {len(json_text)}", level="DEBUG")
                
                try:
                    core.logging.log_event(f"[DeepAgent] Attempting to parse JSON response", level="DEBUG")
                    parsed_response = _recover_json(json_text)
                    
                    # Validation logic moved to _validate_and_execute_tool
                    pass # Fall through to shared validation logic

                except json.JSONDecodeError as e:
                    core.logging.log_event(f"[DeepAgent] Failed to parse JSON: {e}. Attempting repair...", level="WARNING")
                    # Attempt LLM repair for malformed JSON
                    repaired_response = await self._repair_json_with_llm(response_text, f"JSONDecodeError: {e}")
                    
                    if repaired_response and isinstance(repaired_response, dict):
                         core.logging.log_event("[DeepAgent] Malformed JSON successfully repaired by LLM.", level="INFO")
                         parsed_response = repaired_response
                    else:
                        core.logging.log_event(f"[DeepAgent] JSON repair failed. Raw response: {response_text[:500]}", level="ERROR")
                        return f"Error: DeepAgent generated invalid JSON: {response_text[:200]}"

                # --- Validation and Execution Logic (Shared) ---
                return await self._validate_and_execute_tool(parsed_response)
            else:
                core.logging.log_event("The vLLM server returned an empty or invalid response.", level="ERROR")
                return "Error: The vLLM server returned an empty or invalid response."
        except httpx.RequestError as e:
            error_message = f"Error communicating with vLLM server: {e}"
            core.logging.log_event(error_message, level="ERROR")
            return error_message
