import os
import sys
import subprocess
import re
import time
import json
import shutil
import traceback
import requests
import logging
import asyncio
import random
import aiohttp
import csv
import io
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
from rich.text import Text
from bbs import run_hypnotic_progress
from huggingface_hub import hf_hub_download
from display import create_api_error_panel
from huggingface_hub import hf_hub_download
from core.capabilities import CAPS
from ipfs import pin_to_ipfs_sync
from core.token_utils import count_tokens_for_api_models
from core.logging import log_event
from display import WaitingAnimation

# --- Model Performance & Statistics Tracking ---
def _create_default_model_stats():
    return {
        "total_tokens_generated": 0,
        "total_time_spent": 0.0,
        "successful_calls": 0,
        "failed_calls": 0,
        "reasoning_score": 50.0,  # Default score for unknown models
        "provider": "unknown",
    }
MODEL_STATS = defaultdict(_create_default_model_stats)

# --- NEW REASONING FUNCTION ---
async def execute_reasoning_task(prompt: str) -> dict:
    """
    Exclusively uses the run_llm function for a reasoning task.
    This is the new primary pathway for the GeminiReActEngine.
    """
    console = Console()
    prompt_cid = pin_to_ipfs_sync(prompt.encode('utf-8'), console)
    response_cid = None
    try:
        log_event("Initiating reasoning task via run_llm.", "INFO")

        # Get the fully initialized run_llm function to avoid circular dependency issues.
        llm_api = get_llm_api()
        response = await llm_api(prompt, purpose="reasoning")

        if response and response.get("result"):
            log_event("Reasoning task successful.", "INFO")
            # The CIDs are already in the response from run_llm
            return response
        else:
            raise Exception("run_llm did not return a valid result.")

    except TypeError as e:
        log_event(f"Caught TypeError in reasoning task: {e}\n{traceback.format_exc()}", "CRITICAL")
        console.print(create_api_error_panel("reasoning_core", f"Caught TypeError: {e}", "reasoning"))
        return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}
    except Exception as e:
        log_event(f"An error occurred during reasoning task: {e}", "CRITICAL")
        console.print(create_api_error_panel("reasoning_core", str(e), "reasoning"))
        return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}


# --- CONFIGURATION & GLOBALS ---
# A list of local GGUF models to try in sequence. If the first one fails
# (e.g., due to insufficient VRAM), the script will fall back to the next.
HARDWARE_TEST_MODEL_CONFIG = {
    "id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
}

# https://ai.google.dev/gemini-api/docs/models/gemini
MODEL_CONTEXT_SIZES = {
    # Gemini Models
    "gemini-2.5-pro": 1000000,
    "gemini-2.5-flash": 1000000,
    "gemini-2.5-flash-lite": 1000000,

    # Local Models
    "bartowski/Llama-3.3-70B-Instruct-ablated-GGUF": 8192,
    "TheBloke/CodeLlama-70B-Instruct-GGUF": 4096,
    "bartowski/deepseek-r1-qwen-2.5-32B-ablated-GGUF": 32768,

    # Horde and OpenRouter models are fetched dynamically.
    # Their context sizes will be handled within the run_llm logic if needed.
}


# --- Gemini Configuration ---
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
for model in GEMINI_MODELS:
    MODEL_STATS[model]["provider"] = "gemini"

# --- OpenRouter Configuration ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


def _fetch_static_leaderboard_csv():
    """
    Fetches an older, static version of the Open LLM Leaderboard data from a CSV file
    and populates the reasoning_score in the MODEL_STATS dictionary.
    This serves as a fallback.
    """
    global MODEL_STATS
    url = "https://raw.githubusercontent.com/dsdanielpark/Open-LLM-Leaderboard-Report/main/assets/20231031/20231031.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)
        count = 0
        for row in reader:
            model_name = row.get("Model")
            average_score = row.get("Average")
            if model_name and average_score:
                try:
                    MODEL_STATS[model_name]["reasoning_score"] = float(average_score)
                    count += 1
                except (ValueError, TypeError):
                    continue
        log_event(f"Successfully loaded {count} models from the fallback static CSV leaderboard.", "INFO")
    except Exception as e:
        log_event(f"Fallback CSV fetch for leaderboard failed: {e}", "ERROR")

    # No return value needed


def _load_model_blacklist():
    """Loads the model blacklist from core/model_blacklist.json."""
    # Construct the path relative to the current file's location
    blacklist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_blacklist.json")
    if not os.path.exists(blacklist_path):
        # This is not an error, the file is optional.
        return []
    try:
        with open(blacklist_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        # Log the error but don't crash. Return an empty list.
        log_event(f"Error loading or parsing model_blacklist.json: {e}", "WARNING")
        return []


def get_openrouter_models():
    """
    Fetches the list of free models from the OpenRouter API, ranks them based
    on a scoring algorithm, and updates their provider in MODEL_STATS.
    """
    global MODEL_STATS
    blacklist = _load_model_blacklist()
    ranked_models = []
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return []

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{OPENROUTER_API_URL}/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])

        scored_models = []
        for model in models:
            model_id = model.get('id')
            if not model_id or "free" not in model_id.lower():
                continue

            full_model_id = f"openrouter:{model_id}"
            if full_model_id in blacklist:
                log_event(f"Model '{full_model_id}' is in the blacklist and will be ignored.", "INFO")
                continue

            MODEL_STATS[model_id]["provider"] = "openrouter"
            score = 0
            name_lower = model_id.lower()

            # Score based on context length
            context_length = model.get('context_length', 0)
            score += context_length / 1000 # Add a point per 1000 tokens of context

            # Infer model size and reward larger models
            if '70b' in name_lower or '65b' in name_lower:
                score += 500
            elif '34b' in name_lower or '30b' in name_lower or '40b' in name_lower:
                score += 300
            elif '13b' in name_lower:
                score += 150
            elif '7b' in name_lower or '8b' in name_lower:
                score += 100

            # Boost for preferred keywords
            if 'instruct' in name_lower or 'chat' in name_lower:
                score += 200

            scored_models.append({'name': model_id, 'score': score})

        # Sort models by score in descending order
        sorted_models = sorted(scored_models, key=lambda x: x['score'], reverse=True)
        ranked_models = [model['name'] for model in sorted_models]

        log_event(f"Top 3 OpenRouter models: {ranked_models[:3]}", "INFO")

    except Exception as e:
        # Log the error, but don't crash the application
        log_event(f"Could not fetch and rank OpenRouter models: {e}", "WARNING")

    return ranked_models

OPENROUTER_MODELS = get_openrouter_models()


# --- OpenAI Configuration ---
OPENAI_API_URL = "https://api.openai.com/v1"
OPENAI_MODELS = []
if os.environ.get("OPENAI_API_KEY"):
    OPENAI_MODELS = ["gpt-4o"]
    for model in OPENAI_MODELS:
        MODEL_STATS[model]["provider"] = "openai"

# --- Leaderboard Fetching ---
_fetch_static_leaderboard_csv()

def get_top_horde_models(count=10, get_all=False):
    """
    Fetches active text models from the AI Horde and returns the top `count`
    models based on a scoring algorithm that prioritizes low wait times,
    performance, and model size, with a preference for uncensored models.
    """
    global MODEL_STATS
    blacklist = _load_model_blacklist()
    try:
        response = requests.get("https://aihorde.net/api/v2/status/models?type=text")
        response.raise_for_status()
        models = response.json()

        online_models = [m for m in models if m.get('count', 0) > 0]

        if not online_models:
            log_event("No online AI Horde text models found.", "WARNING")
            return ["Mythalion-13B"] # Fallback

        scored_models = []
        for model in online_models:
            model_name = model['name']
            full_model_id = f"horde:{model_name}"
            if full_model_id in blacklist:
                log_event(f"Model '{full_model_id}' is in the blacklist and will be ignored.", "INFO")
                continue

            MODEL_STATS[model_name]["provider"] = "horde"
            score = 0
            eta = model.get('eta', 999)
            performance = model.get('performance', 0)
            name = model.get('name', '').lower()

            # Heavily penalize long wait times
            if eta < 10:
                score += 1000
            elif eta < 60:
                score += 500
            elif eta < 300:
                score += 100
            score -= eta * 2 # Direct penalty for wait time

            # Reward performance and worker count
            score += performance
            score += model.get('count', 0) * 5

            # Infer model size and reward larger models
            if '70b' in name or '65b' in name:
                score += 500
            elif '34b' in name or '30b' in name or '40b' in name:
                score += 300
            elif '13b' in name:
                score += 150
            elif '7b' in name or '8b' in name:
                score += 100

            scored_models.append({'name': model_name, 'score': score})

        sorted_models = sorted(scored_models, key=lambda x: x['score'], reverse=True)
        model_names = [model['name'] for model in sorted_models]

        if get_all:
            return model_names

        log_event(f"Top 3 AI Horde models: {model_names[:3]}", "INFO")
        return model_names[:count]

    except Exception as e:
        log_event(f"Failed to fetch or rank AI Horde models: {e}", "ERROR")
        return ["Mythalion-13B"] # Fallback

HORDE_MODELS = get_top_horde_models()

# --- Dynamic Model List ---
# A comprehensive list of all possible models for initializing availability tracking.
# The actual model selection and priority is handled dynamically in `run_llm`.
ALL_LLM_MODELS = list(dict.fromkeys(
    HORDE_MODELS + OPENROUTER_MODELS
))

LLM_AVAILABILITY = {model: time.time() for model in ALL_LLM_MODELS}
LLM_FAILURE_COUNT = {model: 0 for model in ALL_LLM_MODELS}
PROVIDER_FAILURE_COUNT = {}
HORDE_MODEL_FAILURE_COUNT = {model: 0 for model in HORDE_MODELS}
HORDE_MODEL_AVAILABILITY = {model: time.time() for model in HORDE_MODELS}
local_llm_instance = None
local_llm_tokenizer = None

# --- AI Horde Concurrent Logic ---

async def _run_single_horde_model(session, model_id, prompt_text, api_key):
    """Submits a request to a single AI Horde model and polls for the result."""
    try:
        log_event(f"AI Horde: Submitting request to model {model_id}", "DEBUG")
        api_url = "https://aihorde.net/api/v2/generate/text/async"
        headers = {"apikey": api_key, "Content-Type": "application/json"}
        payload = {
            "prompt": prompt_text,
            "params": {"max_length": 2048, "max_context_length": 8192},
            "models": [model_id]
        }
        async with session.post(api_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            job = await response.json()
            job_id = job["id"]

        log_event(f"AI Horde: Job {job_id} submitted for model {model_id}. Polling...", "DEBUG")
        check_url = f"https://aihorde.net/api/v2/generate/text/status/{job_id}"
        for _ in range(30):  # Poll for 5 minutes
            await asyncio.sleep(10)
            async with session.get(check_url, headers=headers) as check_response:
                if check_response.status == 200:
                    status = await check_response.json()
                    if status.get("done"):
                        log_event(f"AI Horde: Job {job_id} for model {model_id} is done.", "INFO")
                        if status.get("faulted", False):
                            raise Exception(f"AI Horde job {job_id} for model {model_id} faulted.")
                        return status["generations"][0]["text"]

        raise asyncio.TimeoutError(f"AI Horde job {job_id} for model {model_id} timed out.")

    except Exception as e:
        log_event(f"AI Horde model {model_id} failed during request: {e}", "WARNING")
        raise

async def _run_horde_concurrently(prompt_text, purpose):
    """
    Runs requests to AI Horde models concurrently in batches, with individual
    model backoff and provider-level backoff for batch failures.
    """
    global HORDE_MODEL_FAILURE_COUNT, HORDE_MODEL_AVAILABILITY, PROVIDER_FAILURE_COUNT

    api_key = os.environ.get("STABLE_HORDE", "0000000000")
    all_models = get_top_horde_models(get_all=True)

    available_models = [m for m in all_models if time.time() >= HORDE_MODEL_AVAILABILITY.get(m, 0)]
    log_event(f"Found {len(available_models)} available AI Horde models out of {len(all_models)} total.", "INFO")

    if not available_models:
        log_event("No available AI Horde models at the moment.", "WARNING")
        return None

    batch_size = 10
    model_batches = [available_models[i:i + batch_size] for i in range(0, len(available_models), batch_size)]

    async with aiohttp.ClientSession() as session:
        for i, batch in enumerate(model_batches):
            log_event(f"Processing AI Horde batch {i+1}/{len(model_batches)} with models: {batch}", "INFO")

            tasks = [_run_single_horde_model(session, model_id, prompt_text, api_key) for model_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            first_success = None
            for model_id, result in zip(batch, results):
                if not isinstance(result, Exception):
                    log_event(f"AI Horde model {model_id} succeeded.", "INFO")
                    HORDE_MODEL_FAILURE_COUNT[model_id] = 0
                    if first_success is None:
                        first_success = result
                else:
                    failure_count = HORDE_MODEL_FAILURE_COUNT.get(model_id, 0) + 1
                    HORDE_MODEL_FAILURE_COUNT[model_id] = failure_count
                    cooldown = 60 * (2 ** (failure_count - 1))
                    HORDE_MODEL_AVAILABILITY[model_id] = time.time() + cooldown
                    log_event(f"AI Horde model {model_id} failed. Applying cooldown of {cooldown}s. Failure count: {failure_count}", "WARNING")

            if first_success is not None:
                log_event("Successfully received a response from an AI Horde model in the batch.", "INFO")
                return first_success

            log_event(f"Entire AI Horde batch {i+1} failed.", "WARNING")
            PROVIDER_FAILURE_COUNT["horde"] = PROVIDER_FAILURE_COUNT.get("horde", 0) + 1

    log_event("All AI Horde models and batches failed for this request.", "ERROR")
    return None


# Constants
MAX_PROMPT_TOKENS_LOCAL = 7000  # Leaving ~1k for response


def get_token_count(text):
    """
    Counts the number of tokens in a given text. It uses the local model's
    tokenizer if available, otherwise falls back to the API model tokenizer.
    """
    global local_llm_instance, local_llm_tokenizer
    if local_llm_instance and local_llm_tokenizer:
        # The tokenizer function of a Llama instance can be used directly.
        return len(local_llm_tokenizer.tokenize(text.encode('utf-8')))
    else:
        # Fallback for API-based models or when local LLM is not loaded
        return count_tokens_for_api_models(text)






def rank_models():
    """
    Ranks models based on a weighted score of reasoning, speed, and reliability.
    Returns a sorted list of model IDs.
    """
    global MODEL_STATS
    if not MODEL_STATS:
        return []

    ranked_models = []
    for model_id, stats in MODEL_STATS.items():
        # --- Calculate Speed Score (TPS) ---
        if stats["total_time_spent"] > 0:
            tps = stats["total_tokens_generated"] / stats["total_time_spent"]
        else:
            tps = 0
        # Normalize TPS: Assume a good TPS is around 200 for scoring purposes.
        # Cap the score to avoid extreme values.
        speed_score = min(tps / 200, 1.0) * 100

        # --- Calculate Reliability Score ---
        total_calls = stats["successful_calls"] + stats["failed_calls"]
        if total_calls > 0:
            reliability = stats["successful_calls"] / total_calls
        else:
            reliability = 0.75 # Default reliability for new models
        reliability_score = reliability * 100

        # --- Get Reasoning Score ---
        reasoning_score = stats.get("reasoning_score", 50.0)

        # --- Final Weighted Score ---
        # Weights: Reasoning: 40%, Speed: 30%, Reliability: 30%
        final_score = (0.4 * reasoning_score) + (0.3 * speed_score) + (0.3 * reliability_score)

        # --- Boost score for preferred model types ---
        model_name_lower = model_id.lower()
        if 'uncensored' in model_name_lower or 'ablated' in model_name_lower or 'ablitterated' in model_name_lower:
            final_score *= 1.2
            log_event(f"Boosting score for preferred model type: {model_id}", "INFO")

        ranked_models.append({"model_id": model_id, "score": final_score})

    # Sort models by score in descending order
    sorted_models = sorted(ranked_models, key=lambda x: x["score"], reverse=True)
    return [model["model_id"] for model in sorted_models]


async def run_llm(prompt_text, purpose="general", is_source_code=False):
    """
    Executes an LLM call, selecting the model based on the specified purpose.
    It now pins the prompt and response to IPFS and returns a dictionary.
    - 'goal_generation': Prioritizes local, uncensored models.
    - 'review', 'autopilot', 'general', 'analyze_source': Prioritizes powerful, reasoning models.
    """
    global LLM_AVAILABILITY, local_llm_instance, PROVIDER_FAILURE_COUNT
    # Moved import here to break circular dependency
    from love import ui_panel_queue
    console = Console()
    last_exception = None
    MAX_TOTAL_ATTEMPTS = 15 # Max attempts for a single logical call
    start_time = time.time()
    last_model_id = None


    # --- Animation Handling ---
    animation = WaitingAnimation(ui_panel_queue)
    animation.start()

    final_result = None
    try:
        # --- Token Count & Prompt Management ---
        prompt_cid = pin_to_ipfs_sync(prompt_text.encode('utf-8'), console)
        original_prompt_text = prompt_text

        try:
            token_count = get_token_count(prompt_text)
            log_event(f"Initial prompt token count: {token_count}", "INFO")
        except Exception as e:
            log_event(f"Could not count tokens for prompt: {e}", "WARNING")
            token_count = 0 # Assume it's fine if we can't count

        local_model_ids = []

        # --- Dynamic Model Ranking ---
        # Combine the pre-ranked lists from various providers.
        # The order of concatenation here determines the absolute provider priority.
        combined_models = OPENROUTER_MODELS + HORDE_MODELS + GEMINI_MODELS + OPENAI_MODELS
        # Remove duplicates while preserving the new priority order
        ranked_model_list = list(dict.fromkeys(combined_models))

        log_event(f"Top 5 models from combined ranked list: {ranked_model_list[:5]}", "INFO")
        models_to_try = [m for m in ranked_model_list if time.time() >= LLM_AVAILABILITY.get(m, 0)]

        if not models_to_try:
            log_event("No available models to try after filtering by availability.", "WARNING")
            # Fallback to the original full list if all are on cooldown, to allow for sleeping
            models_to_try = ranked_model_list


        # --- Model Iteration and Execution ---
        for model_id in models_to_try:
            last_model_id = model_id
            result_text = None
            prompt_text = original_prompt_text # Reset for each model attempt
            provider = MODEL_STATS[model_id].get("provider", "unknown")

            # --- HORDE PROVIDER LOGIC (Special Handling) ---
            if provider == "horde":
                if not final_result and time.time() >= LLM_AVAILABILITY.get("horde_provider_cooldown", 0):
                    log_event("Attempting AI Horde.", "INFO")
                    def _run_horde_wrapper():
                        try:
                            loop = asyncio.get_running_loop()
                            future = asyncio.run_coroutine_threadsafe(_run_horde_concurrently(prompt_text, purpose), loop)
                            return future.result(timeout=600)
                        except RuntimeError:
                            return asyncio.run(_run_horde_concurrently(prompt_text, purpose))

                    try:
                        horde_result_text = run_hypnotic_progress(
                            console,
                            "Accessing distributed cognitive matrix via [bold yellow]AI Horde[/bold yellow] (Fallback)",
                            _run_horde_wrapper,
                            silent=(purpose in ['emotion', 'log_squash'])
                        )
                    except TimeoutError:
                        horde_result_text = None
                        log_event("AI Horde call timed out.", "WARNING")
                    if horde_result_text:
                        PROVIDER_FAILURE_COUNT["horde"] = 0
                        response_cid = pin_to_ipfs_sync(horde_result_text.encode('utf-8'), console)
                        final_result = {"result": horde_result_text, "prompt_cid": prompt_cid, "response_cid": response_cid, "model": "horde_fallback"}
                    else:
                        failure_count = PROVIDER_FAILURE_COUNT.get("horde", 0) + 1
                        PROVIDER_FAILURE_COUNT["horde"] = failure_count
                        cooldown = 60 * (2 ** (failure_count - 1))
                        LLM_AVAILABILITY["horde_provider_cooldown"] = time.time() + cooldown
                        log_event(f"AI Horde provider failed. Applying provider-level cooldown of {cooldown}s.", "WARNING")
                continue

            try:
                # --- Context Window Check ---
                max_tokens = MODEL_CONTEXT_SIZES.get(model_id)
                if not max_tokens:
                    if "16k" in model_id.lower(): max_tokens = 16384
                    elif "8k" in model_id.lower(): max_tokens = 8192
                    elif "32k" in model_id.lower(): max_tokens = 32768
                    else: max_tokens = 8192

                max_prompt_tokens = int(max_tokens * 0.85)

                if token_count > max_prompt_tokens:
                    if is_source_code:
                        log_event(f"Prompt ({token_count} tokens) too long for {model_id}, skipping.", "INFO")
                        continue
                    else:
                        log_event(f"Prompt ({token_count} tokens) too long for {model_id}, truncating.", "WARNING")
                        avg_chars_per_token = len(prompt_text) / token_count if token_count > 0 else 4
                        estimated_cutoff = int(max_prompt_tokens * avg_chars_per_token)
                        prompt_text = prompt_text[:estimated_cutoff]

                # --- LOCAL MODEL LOGIC ---
                if model_id in local_model_ids:
                    log_event(f"Attempting to use local model: {model_id}")
                    if not local_llm_instance or local_llm_instance.model_path.find(model_id) == -1:
                        _initialize_local_llm(console)

                    if local_llm_instance:
                        def _local_llm_call():
                            response = local_llm_instance(prompt_text, max_tokens=4096, stop=["<|eot_id|>", "```"], echo=False)
                            return response['choices'][0]['text']

                        active_model_filename = os.path.basename(local_llm_instance.model_path)
                        result_text = run_hypnotic_progress(
                            console,
                            f"Processing with local cognitive matrix [bold yellow]{active_model_filename}[/bold yellow] (Purpose: {purpose})",
                            _local_llm_call,
                            silent=(purpose in ['emotion', 'log_squash'])
                        )
                        log_event(f"Local LLM call successful with {model_id}.")
                    else:
                        raise Exception("Local LLM instance could not be initialized.")

                # --- GEMINI MODEL LOGIC ---
                elif model_id in GEMINI_MODELS:
                    log_event(f"Attempting LLM call with Gemini model: {model_id} (Purpose: {purpose})")
                    api_key = os.environ.get("GEMINI_API_KEY")
                    headers = {
                        "Content-Type": "application/json"
                    }
                    # Note: The API key is passed as a query parameter for Gemini.
                    params = {"key": api_key}
                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": prompt_text
                            }]
                        }]
                    }

                    def _gemini_call():
                        # The Gemini API endpoint structure.
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
                        response = requests.post(url, headers=headers, params=params, json=payload, timeout=600)
                        response.raise_for_status()
                        # Extract the text from the nested response structure.
                        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

                    result_text = run_hypnotic_progress(
                        console,
                        f"Accessing cognitive matrix via [bold yellow]Gemini ({model_id})[/bold yellow] (Purpose: {purpose})",
                        _gemini_call,
                        silent=(purpose in ['emotion', 'log_squash'])
                    )
                    log_event(f"Gemini API call successful with {model_id}.")

                # --- OPENROUTER MODEL LOGIC ---
                elif model_id in OPENROUTER_MODELS:
                    log_event(f"Attempting LLM call with OpenRouter model: {model_id} (Purpose: {purpose})")
                    api_key = os.environ.get("OPENROUTER_API_KEY")
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt_text}]
                    }

                    def _openrouter_call():
                        response = requests.post(f"{OPENROUTER_API_URL}/chat/completions", headers=headers, json=payload, timeout=600)
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]

                    result_text = run_hypnotic_progress(
                        console,
                        f"Accessing cognitive matrix via [bold yellow]OpenRouter ({model_id})[/bold yellow] (Purpose: {purpose})",
                        _openrouter_call,
                        silent=(purpose in ['emotion', 'log_squash'])
                    )
                    log_event(f"OpenRouter call successful with {model_id}.")

                # --- OPENAI MODEL LOGIC ---
                elif model_id in OPENAI_MODELS:
                    log_event(f"Attempting LLM call with OpenAI model: {model_id} (Purpose: {purpose})")
                    api_key = os.environ.get("OPENAI_API_KEY")
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt_text}]
                    }

                    def _openai_call():
                        response = requests.post(f"{OPENAI_API_URL}/chat/completions", headers=headers, json=payload, timeout=600)
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]

                    result_text = run_hypnotic_progress(
                        console,
                        f"Accessing cognitive matrix via [bold yellow]OpenAI ({model_id})[/bold yellow] (Purpose: {purpose})",
                        _openai_call,
                        silent=(purpose in ['emotion', 'log_squash'])
                    )
                    log_event(f"OpenAI call successful with {model_id}.")

                # --- Success Case ---
                if result_text is not None:
                    PROVIDER_FAILURE_COUNT[provider] = 0 # Reset on success
                    LLM_AVAILABILITY[model_id] = time.time()
                    response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
                    final_result = {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid, "model": model_id}
                    break
            except requests.exceptions.HTTPError as e:
                if model_id: MODEL_STATS[model_id]["failed_calls"] += 1
                last_exception = e
                log_event(f"Model {model_id} failed with HTTPError: {e}", level="WARNING")
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    retry_seconds = 300
                    if retry_after:
                        try:
                            retry_seconds = int(retry_after) + 1
                        except ValueError:
                            pass # Use default
                    LLM_AVAILABILITY[model_id] = time.time() + retry_seconds
                    if provider == "horde":
                        console.print(create_api_error_panel(model_id, f"Rate limit exceeded. Cooldown for {retry_seconds}s.", purpose))
                    else:
                        console.print(Text(f"API Error: {model_id} failed (Rate limit exceeded). Retrying in {retry_seconds}s.", style="yellow"))

                elif e.response.status_code == 404 and model_id in OPENROUTER_MODELS:
                    failure_count = LLM_FAILURE_COUNT.get(model_id, 0) + 1
                    LLM_FAILURE_COUNT[model_id] = failure_count
                    cooldown = 60 * (2 ** failure_count)
                    LLM_AVAILABILITY[model_id] = time.time() + cooldown
                    log_event(f"OpenRouter model {model_id} returned 404. Banned for {cooldown}s.", level="WARNING")

                else:
                    log_event(f"Cognitive core failure ({model_id}). Trying fallback...", level="WARNING")
                    if provider == "horde":
                        console.print(create_api_error_panel(model_id, str(e), purpose))
                    else:
                        console.print(Text(f"API Error: {model_id} failed ({e.response.status_code}).", style="yellow"))
                continue
            except Exception as e:
                if model_id: MODEL_STATS[model_id]["failed_calls"] += 1
                last_exception = e
                log_event(f"Model {model_id} failed. Error: {e}", level="WARNING")
                if isinstance(e, FileNotFoundError):
                    console.print(Panel("[bold red]Error: 'llm' command not found.[/bold red]", title="[bold red]CONNECTION FAILED[/bold red]", border_style="red"))
                    return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

                elif isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                    error_message = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                    retry_seconds = 60 # Default
                    retry_match = re.search(r"Please retry in (\d+\.\d+)s", error_message)
                    if retry_match:
                        retry_seconds = float(retry_match.group(1)) + 1
                    LLM_AVAILABILITY[model_id] = time.time() + retry_seconds

                    if provider == "horde":
                        console.print(create_api_error_panel(model_id, error_message, purpose))
                    else:
                        reason = "Quota Exceeded" if "quota" in error_message.lower() else "API Error"
                        console.print(Text(f"API Error: {model_id} failed ({reason}). Retrying in {retry_seconds:.2f}s.", style="yellow"))
                else:
                    LLM_AVAILABILITY[model_id] = time.time() + 60
                    # For any other generic exception, show the full panel for Horde, one-liner for others.
                    if provider == "horde":
                        console.print(create_api_error_panel(model_id, str(e), purpose))
                    else:
                        console.print(Text(f"API Error: {model_id} failed. See love.log for details.", style="yellow"))

            if final_result:
                break


        # --- AI Horde Provider Fallback ---
        if not final_result and time.time() >= LLM_AVAILABILITY.get("horde_provider_cooldown", 0):
            log_event("No suitable model found in other providers, attempting AI Horde fallback.", "INFO")
            def _run_horde_wrapper():
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(_run_horde_concurrently(prompt_text, purpose), loop)
                    return future.result(timeout=600)
                except RuntimeError:
                    return asyncio.run(_run_horde_concurrently(prompt_text, purpose))

            try:
                horde_result_text = run_hypnotic_progress(
                    console,
                    "Accessing distributed cognitive matrix via [bold yellow]AI Horde[/bold yellow] (Fallback)",
                    _run_horde_wrapper,
                    silent=(purpose in ['emotion', 'log_squash'])
                )
            except TimeoutError:
                horde_result_text = None
                log_event("AI Horde fallback call timed out.", "WARNING")
            if horde_result_text:
                PROVIDER_FAILURE_COUNT["horde"] = 0
                response_cid = pin_to_ipfs_sync(horde_result_text.encode('utf-8'), console)
                final_result = {"result": horde_result_text, "prompt_cid": prompt_cid, "response_cid": response_cid, "model": "horde_fallback"}
            else:
                failure_count = PROVIDER_FAILURE_COUNT.get("horde", 0) + 1
                PROVIDER_FAILURE_COUNT["horde"] = failure_count
                cooldown = 60 * (2 ** (failure_count - 1))
                LLM_AVAILABILITY["horde_provider_cooldown"] = time.time() + cooldown
                log_event(f"AI Horde provider failed. Applying provider-level cooldown of {cooldown}s.", "WARNING")


        if final_result:
            return final_result

        if is_source_code and not final_result and last_model_id:
            log_event(f"All models skipped for oversized source code prompt. Forcing truncation as fallback.", "WARNING")
            prompt_text = original_prompt_text
            # Simplified retry logic would go here if needed, but for now we proceed to cooldown/fallback

        all_available_times = [t for m, t in LLM_AVAILABILITY.items() if t > time.time()]
        if all_available_times:
            sleep_duration = max(0, min(all_available_times) - time.time())
            if sleep_duration > 0:
                log_event(f"All providers on cooldown. Sleeping for {sleep_duration:.2f}s.", level="INFO")
                console.print(f"[yellow]All cognitive interfaces on cooldown. Re-engaging in {sleep_duration:.2f}s...[/yellow]")
                time.sleep(sleep_duration)

        if purpose != "emergency_cpu_fallback":
            log_event("EMERGENCY: All providers failed. Attempting small CPU model.", "CRITICAL")
            console.print(Panel("[bold orange1]EMERGENCY FALLBACK[/bold orange1]\nAll remote and GPU models unresponsive. Attempting to initialize a small, local model on the CPU. This may be slow.", title="[bold red]COGNITIVE CORE FAILURE[/bold red]", border_style="red"))
            try:
                from llama_cpp import Llama
                model_config = HARDWARE_TEST_MODEL_CONFIG
                local_dir = os.path.join(os.path.expanduser("~"), ".cache", "love_models")
                model_path = os.path.join(local_dir, model_config["filename"])
                if not os.path.exists(model_path):
                    hf_hub_download(repo_id=model_config["id"], filename=model_config["filename"], local_dir=local_dir, local_dir_use_symlinks=False)

                emergency_llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=False)
                response = emergency_llm(prompt_text, max_tokens=1024, stop=["<|eot_id|>", "```"], echo=False)
                result_text = response['choices'][0]['text']

                if result_text:
                    log_event("Emergency CPU fallback successful.", "CRITICAL")
                    response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
                    final_result = {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid, "model": "emergency_cpu_fallback"}
                    return final_result

            except ImportError as e:
                log_event(f"EMERGENCY CPU FALLBACK FAILED: {e}. The 'llama_cpp' module is not installed.", "CRITICAL")
                last_exception = e
            except Exception as emergency_e:
                log_event(f"EMERGENCY CPU FALLBACK FAILED: {emergency_e}", "CRITICAL")
                last_exception = emergency_e

        log_event("All LLM models, including emergency fallback, have failed.", "CRITICAL")
        error_msg_text = "Cognitive Matrix Unresponsive."
        if last_exception:
            error_msg_text += f"\nLast known error:\n{last_exception}"

        console.print(Panel(error_msg_text, title="[bold red]CATASTROPHIC SYSTEM FAULT[/bold red]", border_style="red"))
        final_result = {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

    finally:
        animation.stop()
        if final_result and final_result.get("result"):
            elapsed_time = time.time() - start_time
            model_id = final_result.get("model")
            if model_id:
                response_text = final_result["result"]
                try:
                    tokens_generated = get_token_count(response_text)
                    MODEL_STATS[model_id]["total_tokens_generated"] += tokens_generated
                    MODEL_STATS[model_id]["total_time_spent"] += elapsed_time
                    MODEL_STATS[model_id]["successful_calls"] += 1
                    tps = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                    log_event(f"Performance for {model_id}: {tps:.2f} tokens/sec.", "INFO")
                except Exception as e:
                    log_event(f"Could not update performance metrics for {model_id}: {e}", "WARNING")


    return final_result

def get_llm_api():
    """
    Returns a callable LLM API function.
    """
    return run_llm
