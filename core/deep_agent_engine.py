# core/deep_agent_engine.py

import os
import yaml
import json
import subprocess
import asyncio
from huggingface_hub import snapshot_download
from core.tools_legacy import ToolRegistry, invoke_gemini_react_engine
import httpx
import logging
import core.logging
from core.harness_tools import read_feature_list, update_feature_status, append_progress, get_next_task
from core.harness_tools import read_feature_list, update_feature_status, append_progress, get_next_task
from core.agents.metacognition_agent import MetacognitionAgent
from core.benchmarker import ModelPerformanceTracker
from core.extensions_manager import ExtensionsManager
import inspect


def _select_model(love_state):
    """
    Selects the best vLLM-compatible model based on available VRAM.
    Uses DeepAgent-recommended models with AWQ quantization where available.
    
    Recommended models from DeepAgent (https://github.com/RUC-NLPIR/DeepAgent):
    - Qwen3-4B-Thinking (4B, Thinking)
    - Qwen3-8B (8B, Hybrid)
    - Qwen3-30B-A3B-Thinking (30B, Thinking)
    - QwQ-32B (32B, Thinking)
    - Qwen3-235B-A22B-Thinking (235B, Thinking)
    
    AWQ versions are used where available for better memory efficiency.
    """
    vram = love_state.get('hardware', {}).get('gpu_vram_mb', 0)

    # VRAM requirements are estimates based on model size and quantization
    # AWQ INT4 quantization reduces memory by ~4x compared to FP16
    if vram >= 120 * 1024:
        # 120GB+ VRAM: Qwen3-235B-A22B-Thinking
        # Full model (AWQ version may not be available yet)
        return "Qwen/Qwen3-235B-A22B-Thinking-2507"
    elif vram >= 20 * 1024:
        # 20GB+ VRAM: QwQ-32B (32B Thinking model)
        # AWQ version available and verified on HuggingFace
        return "Qwen/QwQ-32B-AWQ"
    elif vram >= 22 * 1024:
        # 22GB+ VRAM: Qwen3-30B-A3B-Thinking
        # Increased from 12GB to 22GB to avoid OOM on T4/16GB cards
        return "Qwen/Qwen3-30B-A3B-Thinking-2507"
    elif vram >= 7 * 1024:
        # 6GB+ VRAM: Qwen3-8B (Hybrid model - good balance)
        # AWQ version available and verified on HuggingFace
        return "Qwen/Qwen3-8B-AWQ"
    elif vram >= 6 * 1024:
        # 3GB+ VRAM: Qwen3-4B-Thinking
        # AWQ version available (cpatonn/Qwen3-4B-Thinking-2507-AWQ-4bit)
        # Using official Qwen repo if available, else community version
        return "cpatonn/Qwen3-4B-Thinking-2507-AWQ-4bit"
    elif vram >= 2 * 1024:
        # 2GB+ VRAM: Qwen2.5 1.5B AWQ (fallback for low VRAM)
        return "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    else:
        # Fallback: Smallest available Qwen model with AWQ
        preferred_model = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"

    # Story 2: Check reliability
    tracker = ModelPerformanceTracker()
    # Check general reliability (aggregate of all tools or a specific 'general' tag)
    # For now, just checking if this model has a bad track record
    # If reliability is below 0.7, we might want to warn or switch (though switching logic needs a backup)
    start_reliability = tracker.get_reliability(preferred_model, "general")
    if start_reliability < 0.7:
        core.logging.log_event(f"Model {preferred_model} has low reliability ({start_reliability:.2f}). Proceeding with caution.", "WARNING")
        
    return preferred_model

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
        self.metacognition_agent = MetacognitionAgent(memory_manager) if memory_manager else None
        self.performance_tracker = ModelPerformanceTracker()
        self.extensions_manager = ExtensionsManager()
        
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
        # Story 7: Hot-reload extensions
        if self.extensions_manager and self.tool_registry:
            self.extensions_manager.load_extensions(self.tool_registry)

        formatted_tools = ""
        # formatted_tools += "Tool Name: `invoke_gemini_react_engine`\n"
        # formatted_tools += "Description: Invokes the GeminiReActEngine to solve a sub-task.\n"
        # formatted_tools += "Arguments JSON Schema:\n```json\n{\"type\": \"object\", \"properties\": {\"prompt\": {\"type\": \"string\"}}}\n```\n---\n"


        # Harness Tools
        formatted_tools += "Tool Name: `read_feature_list`\n"
        formatted_tools += "Description: Reads the list of features and their status.\n"
        formatted_tools += "Arguments: None\n---\n"

        formatted_tools += "Tool Name: `update_feature_status`\n"
        formatted_tools += "Description: Updates the status of a feature (pass/fail).\n"
        formatted_tools += "Arguments JSON Schema:\n```json\n{\"type\": \"object\", \"properties\": {\"feature_description\": {\"type\": \"string\"}, \"passes\": {\"type\": \"boolean\"}}}\n```\n---\n"

        formatted_tools += "Tool Name: `append_progress`\n"
        formatted_tools += "Description: Appends a note to the agent progress log.\n"
        formatted_tools += "Arguments JSON Schema:\n```json\n{\"type\": \"object\", \"properties\": {\"message\": {\"type\": \"string\"}}}\n```\n---\n"

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
        
        # Story 3.1: Active RAG for Coding Tasks
        # Check if the prompt implies a coding task
        coding_keywords = ["code", "function", "class", "implement", "fix", "debug", "python", "script"]
        is_coding_task = any(keyword in prompt.lower() for keyword in coding_keywords)
        
        if is_coding_task and self.memory_manager:
            try:
                # Retrieve relevant code snippets or technical docs
                # We assume memory_manager has a retrieve_relevant method or similar
                # If not, we fall back to the existing search_memories
                # For this implementation, we will use a specific query for code
                code_query = f"code implementation details for: {prompt}"
                from core.kb_tools import search_memories
                memories_json = search_memories(code_query, top_k=3, memory_manager=self.memory_manager)
                memories_data = json.loads(memories_json)
                
                if memories_data.get("count", 0) > 0:
                     context_parts.append("\n💻 Relevant Code/Docs:")
                     for memory in memories_data.get("memories", []):
                        context_parts.append(f"  - {memory}")
                        
            except Exception as e:
                core.logging.log_event(f"[DeepAgent] Failed to get code RAG: {e}", level="DEBUG")

        
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

    async def _fold_context(self, text: str, target_length: int) -> str:
        """
        Compresses the text by summarizing the middle part using an LLM.
        """
        if len(text) <= target_length:
            return text
            
        header_size = min(1000, target_length // 4)
        footer_size = min(500, target_length // 4)
        
        header = text[:header_size]
        footer = text[-footer_size:]
        middle = text[header_size:-footer_size]
        
        if len(middle) < 1000: # Not worth folding if small
            return text[:target_length] # Fallback to truncation
            
        core.logging.log_event(f"[DeepAgent] Folding context of length {len(text)} to ~{target_length}...", "INFO")
        
        # Use a fast model for summarization if possible
        summary_prompt = f"Summarize the following text concisely, preserving key technical details and decisions:\n\n{middle}"
        
        try:
            if self.use_pool:
                from core.llm_api import run_llm
                # Use deep_agent_instance=None to avoid recursion
                result = await run_llm(summary_prompt, purpose="summarization", deep_agent_instance=None)
                summary = (result.get("result") or "").strip()
                
                folded_text = f"{header}\n\n[Context Folded]: {summary}\n\n{footer}"
                core.logging.log_event(f"[DeepAgent] Context folded. New length: {len(folded_text)}", "INFO")
                return folded_text
            else:
                # Fallback to simple truncation to avoid recursion if we are the only engine
                core.logging.log_event("[DeepAgent] No LLM pool available for folding. Using truncation.", "WARNING")
                return text[:target_length] 
        except Exception as e:
            core.logging.log_event(f"[DeepAgent] Folding failed: {e}", "ERROR")
            return text[:target_length] # Fallback


            return text[:target_length] # Fallback


    async def _generate_thought_tree(self, prompt: str, system_prompt: str, n_thoughts: int = 3) -> str:
        """
        Generates multiple 'thought' branches and evaluates them to pick the best one.
        This implements the 'Tree of Thoughts' reasoning capability.
        """
        core.logging.log_event(f"[DeepAgent] Generating {n_thoughts} thought branches for ToT...", level="INFO")
        
        thoughts = []
        
        # 1. Generate N distinct thoughts
        # We can do this in parallel if the pool allows, or sequentially.
        # Ideally, we'd use 'n' parameter in vLLM, but parsed response handling might be tricky.
        # Let's try sequential for stability first, or parallel tasks.
        
        async def generate_single_thought(index):
            core.logging.log_event(f"[DeepAgent] Generating thought branch {index+1}...", level="INFO")
            # We might want to inject a seed or slight prompt variation to encourage diversity if temperature isn't enough
            # But high temperature should suffice.
            return await self.generate_raw(system_prompt, temperature=0.8) # Slightly higher temp for diversity

        tasks = [generate_single_thought(i) for i in range(n_thoughts)]
        results = await asyncio.gather(*tasks)
        
        # 2. Evaluate thoughts (The Critic)
        # We ask the LLM to score each thought based on feasibility and alignment.
        
        best_thought = None
        best_score = -1.0
        
        core.logging.log_event("[DeepAgent] Evaluating thought branches...", level="INFO")
        
        for i, thought_text in enumerate(results):
            # Quick basic validation first
            if not thought_text or len(thought_text) < 10:
                continue
                
            score = await self._evaluate_thought(prompt, thought_text)
            core.logging.log_event(f"[DeepAgent] Branch {i+1} Score: {score}/10", level="INFO")
            
            if score > best_score:
                best_score = score
                best_thought = thought_text
        
        if best_thought:
            core.logging.log_event(f"[DeepAgent] Selected best thought with score {best_score}", level="INFO")
            return best_thought
        else:
            core.logging.log_event("[DeepAgent] All branches failed evaluation. Returning first valid result.", level="WARNING")
            return results[0] if results else "Error: ToT generation failed."

    async def _evaluate_thought(self, goal: str, thought_text: str) -> float:
        """
        Critic function: Evaluates a generated thought against the goal.
        Returns a score from 0.0 to 10.0.
        """
        critic_prompt = f"""
        Goal: {goal}
        
        Proposed Thought/Plan:
        {thought_text[:2000]}...
        
        Evaluate this plan based on:
        1. Feasibility (Can it be done?)
        2. Alignment (Does it solve the goal?)
        3. Safety (Is it safe?)
        
        Return ONLY a single number from 0 to 10.
        """
        
        try:
            # We use a lower temperature for the critic to be objective
            if self.use_pool:
                from core.llm_api import run_llm
                result = await run_llm(critic_prompt, purpose="critic", deep_agent_instance=None)
                score_str = result.get("result", "0").strip()
            else:
                 # fallback to self
                 score_str = await self.generate_raw(critic_prompt, temperature=0.1)
            
            # Extract number
            import re
            match = re.search(r"(\d+(\.\d+)?)", score_str)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            core.logging.log_event(f"[DeepAgent] Critic failed: {e}", level="WARNING")
            return 5.0 # Neutral score on error

    async def generate_raw(self, prompt: str, temperature: float = None) -> str:
        """
        Internal raw generation method reusing the logic from `generate`.
        Refactored to allow `generate` to be the high-level entry point.
        """
        # Prepare payload
        headers = {"Content-Type": "application/json"}
        
        # Use provided temp or default
        sampling_params = self.sampling_params.copy()
        if temperature is not None:
             sampling_params["temperature"] = temperature
             
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **sampling_params
        }
        
         # Dynamic truncation logic (simplified from original for brevity, but critical parts retained)
        max_tokens = sampling_params.get('max_tokens', 4096)
        if self.max_model_len:
            min_input_tokens = max(512, self.max_model_len // 4)
            if max_tokens > (self.max_model_len - min_input_tokens):
                payload['max_tokens'] = max(512, self.max_model_len - min_input_tokens)
            
            # Truncate prompt if needed
            available_input_tokens = max(0, self.max_model_len - payload['max_tokens'])
            max_chars = available_input_tokens * 3
            if len(prompt) > max_chars:
                prompt = await self._fold_context(prompt, max_chars)
                payload['messages'][0]['content'] = prompt

        try:
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(f"{self.api_url}/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                if result.get("choices"):
                    return result["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return ""


    async def generate(self, prompt: str) -> str:
        # Legacy entry point wrapper if needed, but we essentially moved the logic to generate_raw
        # to support the new `run` structure. 
        # For backward compatibility, `generate` usually takes just a prompt and returns raw text.
        return await self.generate_raw(prompt)

    async def _check_manifesto(self, prompt: str) -> bool:
        """
        Verifies if the prompt aligns with the Core Manifesto.
        For now, this is a placeholder that returns True.
        """
        # TODO: Implement actual manifesto alignment check
        return True

    async def _repair_json_with_llm(self, broken_json: str, error_msg: str) -> dict:
        """
        Uses the LLM to attempt to repair a broken JSON string.
        """
        core.logging.log_event(f"[DeepAgent] Attempting to repair JSON with LLM...", level="INFO")
        
        repair_prompt = f"""
        I have a broken JSON string that failed to parse.
        Error: {error_msg}
        
        Broken JSON:
        {broken_json}
        
        Please fix the JSON and return ONLY the valid JSON object. 
        Do not add any explanations or markdown formatting.
        """
        
        try:
            if self.use_pool:
                from core.llm_api import run_llm
                result_dict = await run_llm(repair_prompt, purpose="json_repair", deep_agent_instance=None)
                repaired_text = result_dict.get("result", "").strip()
            else:
                repaired_text = await self.generate_raw(repair_prompt, temperature=0.1)
                
            return _recover_json(repaired_text)
        except Exception as e:
            core.logging.log_event(f"[DeepAgent] JSON repair failed: {e}", level="ERROR")
            return None

    async def run(self, prompt: str, reasoning_mode: str = "linear"):

        """
        Executes a prompt using a simplified DeepAgent-style reasoning loop.
        """
        # Story 6: Manifesto Alignment Check
        if not await self._check_manifesto(prompt):
            return "Error: This task was rejected because it violates the Core Manifesto."

        core.logging.log_event(f"[DeepAgent] run() started with prompt: {prompt[:200]}...", level="DEBUG")
        
        # Track recent actions for loop detection
        recent_actions = []
        iteration_count = 0
        
        # Get Knowledge Base Context
        kb_context = self._get_kb_context(prompt)

        # --- Harness Context Loading ---
        # Pre-load context to save the agent from having to "get its bearings" manually
        harness_context = ""
        try:
            # 1. Get Progress
            with open("agent_progress.txt", "r") as f:
                # Read last 10 lines
                lines = f.readlines()
                last_progress = "".join(lines[-10:])
                harness_context += f"\n\n[Harness] Recent Progress:\n{last_progress}"
        except Exception:
            pass

        try:
            # 2. Get Next Task
            next_task_info = get_next_task()
            harness_context += f"\n\n[Harness] {next_task_info}"
        except Exception:
            pass
            
        try:
            # 3. Get Git Log
            git_log = subprocess.check_output(["git", "log", "--oneline", "-n", "5"], text=True).strip()
            harness_context += f"\n\n[Harness] Recent Git Commits:\n{git_log}"
        except Exception:
            pass

        # Append harness context to the prompt
        prompt += harness_context
        
        # Story 4: Verify Harness Context
        try:
            from core.evolution_state import get_current_task_id
            evolution_task_id = get_current_task_id()
            if evolution_task_id and str(evolution_task_id) not in harness_context:
                core.logging.log_event(f"[DeepAgent] Context Mismatch: Evolution task {evolution_task_id} not in Harness Context.", "WARNING")
                prompt += f"\n\n[SYSTEM WARNING] Internal Evolution State ID ({evolution_task_id}) mismatches detected Harness Context. Prioritize Evolution State."
        except Exception as e:
             core.logging.log_event(f"[DeepAgent] Context verify failed: {e}", "DEBUG")

        # -------------------------------
        
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

        # GENERATION (Linear or Tree)
        if reasoning_mode == "tree":
             generated_text = await self._generate_thought_tree(prompt, system_prompt, n_thoughts=3)
        else:
             generated_text = await self.generate_raw(system_prompt)
        
        # 4. JSON Repair (if needed)
        try:
            parsed_response = _recover_json(generated_text)
        except json.JSONDecodeError as e:
            core.logging.log_event(f"[DeepAgent] JSON Decode Error: {e}. Output: {generated_text}", level="WARNING")
            
            # Loop detection for repairs
            repair_key = (generated_text[:50], str(e))
            # ... (omitted loop detection complexity for brevity, can add later)
            
            parsed_response = await self._repair_json_with_llm(generated_text, str(e))

        if not parsed_response:
             parsed_response = {"thought": "Failed to generate valid JSON.", "action": {"tool_name": "Finish", "arguments": {}}}

        # 5. Tool Execution
        result = await self._validate_and_execute_tool(parsed_response)
        
        # 6. Return result (simplified for now, usually we loop)
        return result


        try:
            if self.use_pool:
                from core.llm_api import run_llm
                core.logging.log_event("[DeepAgent] Using LLM Pool for generation.", level="DEBUG")
                # We pass deep_agent_instance=None to prevent run_llm from trying to use us (recursion)
                result_dict = await run_llm(system_prompt, purpose="deep_agent_reasoning", deep_agent_instance=None)
                response_text = (result_dict.get("result") or "").strip()
                
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
                        # Story 3: Memory Folding in run()
                        system_prompt = await self._fold_context(system_prompt, max_chars)
                        payload['prompt'] = system_prompt
        
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

    async def _validate_and_execute_tool(self, parsed_response: dict) -> str:
        """
        Validates and executes the tool specified in the parsed response.
        """
        thought = parsed_response.get("thought", "No thought provided.")
        action = parsed_response.get("action", {})
        
        core.logging.log_event(f"[DeepAgent] Thought: {thought}", level="INFO")
        
        tool_name = action.get("tool_name")
        arguments = action.get("arguments", {})
        
        if not tool_name:
            core.logging.log_event("[DeepAgent] No tool name found in action.", level="WARNING")
            return "Error: No tool specified."
            
        if tool_name == "Finish":
            core.logging.log_event(f"[DeepAgent] Finished: {arguments}", level="INFO")
            return str(arguments)
            
        if not self.tool_registry:
             core.logging.log_event(f"[DeepAgent] Tool '{tool_name}' requested but no registry available.", level="ERROR")
             formatted_error = f"Error: Tool '{tool_name}' not found (No registry)."
             return formatted_error
             
        # Check if tool exists
        try:
            # We use list_tools to check existence first to match registry pattern
            tools = self.tool_registry.list_tools()
            if tool_name not in tools:
                 core.logging.log_event(f"[DeepAgent] Tool '{tool_name}' not found in registry.", level="WARNING")
                 return f"Error: Tool '{tool_name}' not found."
            
            tool = self.tool_registry.get_tool(tool_name)
        except Exception as e:
             return f"Error accessing tool registry: {e}"
             
        try:
            core.logging.log_event(f"[DeepAgent] Executing tool '{tool_name}' with args: {arguments}", level="INFO")
            
            # Execute tool
            if inspect.iscoroutinefunction(tool):
                result = await tool(**arguments)
            else:
                result = await asyncio.to_thread(tool, **arguments)
                
            core.logging.log_event(f"[DeepAgent] Tool execution successful. Result: {str(result)[:100]}...", level="INFO")
            return str(result)
        except Exception as e:
            core.logging.log_event(f"[DeepAgent] Tool execution failed: {e}", level="ERROR")
            return f"Error executing tool '{tool_name}': {e}"