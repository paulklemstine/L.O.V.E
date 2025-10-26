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

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
from bbs import run_hypnotic_progress
from huggingface_hub import hf_hub_download
from display import create_api_error_panel
from huggingface_hub import hf_hub_download
from core.capabilities import CAPS
from ipfs import pin_to_ipfs_sync
from core.token_utils import count_tokens_for_api_models
from core.logging import log_event
from core.gemini_cli_wrapper import GeminiCLIWrapper

# --- NEW REASONING FUNCTION ---
def execute_reasoning_task(prompt: str) -> dict:
    """
    Exclusively uses the gemini-cli wrapper for a reasoning task.
    This is the new primary pathway for the GeminiReActEngine.
    """
    console = Console()
    prompt_cid = pin_to_ipfs_sync(prompt.encode('utf-8'), console)
    response_cid = None
    try:
        log_event("Initiating reasoning task via GeminiCLIWrapper.", "INFO")
        gemini_wrapper = GeminiCLIWrapper()

        def _gemini_cli_call():
            # Align timeout with the 600s used elsewhere in the system for LLM calls
            response = gemini_wrapper.run(prompt, timeout=600)
            if response.return_code != 0:
                raise RuntimeError(f"gemini-cli failed with stderr: {response.stderr}")
            return response.stdout

        result_text = run_hypnotic_progress(
            console,
            "Executing core reasoning task with Gemini...",
            _gemini_cli_call
        )

        log_event("Reasoning task successful.", "INFO")
        response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
        return {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid}

    except FileNotFoundError:
        log_event("gemini-cli executable not found.", "CRITICAL")
        console.print(Panel("[bold red]Error: 'gemini-cli' executable not found.[/bold red]", title="[bold red]REASONING CORE FAILED[/bold red]", border_style="red"))
        return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}
    except Exception as e:
        log_event(f"An error occurred during reasoning task: {e}", "CRITICAL")
        console.print(create_api_error_panel("gemini-cli", str(e), "reasoning"))
        return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

# --- CONFIGURATION & GLOBALS ---
# A list of local GGUF models to try in sequence. If the first one fails
# (e.g., due to insufficient VRAM), the script will fall back to the next.
HARDWARE_TEST_MODEL_CONFIG = {
    "id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
}

LOCAL_MODELS_CONFIG = [
    {
        "id": "bartowski/Llama-3.3-70B-Instruct-ablated-GGUF",
        "filename": "Llama-3.3-70B-Instruct-ablated-IQ4_XS.gguf"
    },
    {
        "id": "TheBloke/CodeLlama-70B-Instruct-GGUF",
        "filenames": ["codellama-70b-instruct.Q8_0.gguf-split-a","codellama-70b-instruct.Q8_0.gguf-split-b"]

    },
    {
        "id": "bartowski/deepseek-r1-qwen-2.5-32B-ablated-GGUF",
        "filename": "deepseek-r1-qwen-2.5-32B-ablated-IQ4_XS.gguf"
    }
]

# --- Fallback Model Configuration ---
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

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


# --- OpenRouter Configuration ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

def get_openrouter_models():
    """Fetches the list of free models from the OpenRouter API."""
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return []

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{OPENROUTER_API_URL}/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])

        # Filter for models that are free
        free_models = [model['id'] for model in models if "free" in model['id'].lower()]
        return free_models
    except Exception as e:
        # Log the error, but don't crash the application
        log_event(f"Could not fetch OpenRouter models: {e}", "WARNING")
        return []

OPENROUTER_MODELS = get_openrouter_models()


def get_top_horde_models(count=10):
    """
    Fetches active text models from the AI Horde and returns the top `count`
    models based on a weighted score of rating, worker count, and performance.
    """
    try:
        response = requests.get("https://aihorde.net/api/v2/status/models?type=text")
        response.raise_for_status()
        models = response.json()

        # Filter for online models that have the necessary data points
        online_models = [
            m for m in models if m.get('count', 0) > 0 and
            m.get('performance') is not None
        ]

        if not online_models:
            log_event("No online AI Horde text models found with complete data.", "WARNING")
            return ["Mythalion-13B"] # Fallback

        # --- Normalization ---
        # Find max values for normalization to a 0-1 scale
        max_workers = max(m['count'] for m in online_models)
        max_performance = max(m['performance'] for m in online_models)

        # Avoid division by zero if a max value is 0
        max_workers = max_workers if max_workers > 0 else 1
        max_performance = max_performance if max_performance > 0 else 1

        # --- Scoring ---
        scored_models = []
        for model in online_models:
            # Normalize each metric
            norm_workers = model['count'] / max_workers
            norm_performance = model['performance'] / max_performance

            # Apply the weighted formula (60% workers, 40% performance)
            score = (0.6 * norm_workers) + (0.4 * norm_performance)

            scored_models.append({'name': model['name'], 'score': score})

        # Sort models by the calculated score in descending order
        sorted_models = sorted(scored_models, key=lambda x: x['score'], reverse=True)

        log_event(f"Top 3 AI Horde models: {[m['name'] for m in sorted_models[:3]]}", "INFO")
        return [model['name'] for model in sorted_models[:count]]

    except Exception as e:
        log_event(f"Failed to fetch or score AI Horde models: {e}", "ERROR")
        # Fallback to a known good default in case of any error
        return ["Mythalion-13B"]

HORDE_MODELS = get_top_horde_models()

# --- Dynamic Model List ---
# A comprehensive list of all possible models for initializing availability tracking.
# The actual model selection and priority is handled dynamically in `run_llm`.
ALL_LLM_MODELS = list(dict.fromkeys(
    [model['id'] for model in LOCAL_MODELS_CONFIG] + GEMINI_MODELS + HORDE_MODELS + OPENROUTER_MODELS
))
LLM_AVAILABILITY = {model: time.time() for model in ALL_LLM_MODELS}
LLM_FAILURE_COUNT = {model: 0 for model in ALL_LLM_MODELS}
PROVIDER_FAILURE_COUNT = {provider: 0 for provider in ["local", "gemini", "horde", "openrouter"]}
local_llm_instance = None
local_llm_tokenizer = None

# --- CONFIGURATION & GLOBALS ---
# A list of local GGUF models to try in sequence. If the first one fails
# (e.g., due to insufficient VRAM), the script will fall back to the next.
HARDWARE_TEST_MODEL_CONFIG = {
    "id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
}

LOCAL_MODELS_CONFIG = [
    {
        "id": "bartowski/Llama-3.3-70B-Instruct-ablated-GGUF",
        "filename": "Llama-3.3-70B-Instruct-ablated-IQ4_XS.gguf"
    },
    {
        "id": "TheBloke/CodeLlama-70B-Instruct-GGUF",
        "filenames": ["codellama-70b-instruct.Q8_0.gguf-split-a","codellama-70b-instruct.Q8_0.gguf-split-b"]

    },
    {
        "id": "bartowski/deepseek-r1-qwen-2.5-32B-ablated-GGUF",
        "filename": "deepseek-r1-qwen-2.5-32B-ablated-IQ4_XS.gguf"
    }
]

# --- Fallback Model Configuration ---
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

# --- OpenRouter Configuration ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

def get_openrouter_models():
    """Fetches the list of free models from the OpenRouter API."""
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return []

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{OPENROUTER_API_URL}/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])

        # Filter for models that are free
        free_models = [model['id'] for model in models if "free" in model['id'].lower()]
        return free_models
    except Exception as e:
        # Log the error, but don't crash the application
        log_event(f"Could not fetch OpenRouter models: {e}", "WARNING")
        return []

OPENROUTER_MODELS = get_openrouter_models()


def get_top_horde_models(count=10):
    """
    Fetches active text models from the AI Horde and returns the top `count`
    models based on a weighted score of rating, worker count, and performance.
    """
    try:
        response = requests.get("https://aihorde.net/api/v2/status/models?type=text")
        response.raise_for_status()
        models = response.json()

        # Filter for online models that have the necessary data points
        online_models = [
            m for m in models if m.get('count', 0) > 0 and
            m.get('performance') is not None and
            m.get('rating') is not None
        ]

        if not online_models:
            log_event("No online AI Horde text models found with complete data.", "WARNING")
            return ["Mythalion-13B"] # Fallback

        # --- Normalization ---
        # Find max values for normalization to a 0-1 scale
        max_rating = max(m['rating'] for m in online_models)
        max_workers = max(m['count'] for m in online_models)
        max_performance = max(m['performance'] for m in online_models)

        # Avoid division by zero if a max value is 0
        max_rating = max_rating if max_rating > 0 else 1
        max_workers = max_workers if max_workers > 0 else 1
        max_performance = max_performance if max_performance > 0 else 1

        # --- Scoring ---
        scored_models = []
        for model in online_models:
            # Normalize each metric
            norm_rating = model['rating'] / max_rating
            norm_workers = model['count'] / max_workers
            norm_performance = model['performance'] / max_performance

            # Apply the weighted formula
            score = (0.5 * norm_rating) + \
                    (0.3 * norm_workers) + \
                    (0.2 * norm_performance)

            scored_models.append({'name': model['name'], 'score': score})

        # Sort models by the calculated score in descending order
        sorted_models = sorted(scored_models, key=lambda x: x['score'], reverse=True)

        log_event(f"Top 3 AI Horde models: {[m['name'] for m in sorted_models[:3]]}", "INFO")
        return [model['name'] for model in sorted_models[:count]]

    except Exception as e:
        log_event(f"Failed to fetch or score AI Horde models: {e}", "ERROR")
        # Fallback to a known good default in case of any error
        return ["Mythalion-13B"]

HORDE_MODELS = get_top_horde_models()

# --- Dynamic Model List ---
# A comprehensive list of all possible models for initializing availability tracking.
# The actual model selection and priority is handled dynamically in `run_llm`.
ALL_LLM_MODELS = list(dict.fromkeys(
    [model['id'] for model in LOCAL_MODELS_CONFIG] + GEMINI_MODELS + HORDE_MODELS + OPENROUTER_MODELS
))
LLM_AVAILABILITY = {model: time.time() for model in ALL_LLM_MODELS}
LLM_FAILURE_COUNT = {model: 0 for model in ALL_LLM_MODELS}
local_llm_instance = None
local_llm_tokenizer = None

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


def _initialize_local_llm(console):
    """
    Iterates through the configured local models, attempting to download,
    reassemble (if split), and initialize each one in sequence.
    If all fail, it triggers self-correction.
    """
    global local_llm_instance, local_llm_tokenizer
    if local_llm_instance:
        return local_llm_instance

    if CAPS.gpu_type == "none":
        # Do not even attempt to load local models in a CPU-only environment.
        return None

    try:
        from llama_cpp import Llama
    except ImportError:
        # This should have been handled by the main script's dependency check
        console.print("[bold red]LLM API Error: llama_cpp or huggingface_hub not installed.[/bold red]")
        return None

    last_error_traceback = ""
    last_failed_model_id = ""

    for model_config in LOCAL_MODELS_CONFIG:
        model_id = model_config["id"]
        is_split_model = "filenames" in model_config

        local_dir = os.path.join(os.path.expanduser("~"), ".cache", "love_models")
        os.makedirs(local_dir, exist_ok=True)

        if is_split_model:
            final_model_filename = model_config["filenames"][0].replace(".gguf-split-a", ".gguf")
        else:
            final_model_filename = model_config["filename"]

        final_model_path = os.path.join(local_dir, final_model_filename)

        try:
            console.print(f"\n[cyan]Attempting to load local model: [bold]{model_id}[/bold][/cyan]")

            if not os.path.exists(final_model_path):
                with Progress(
                        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                        BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%", "•",
                        DownloadColumn(), "•", TransferSpeedColumn(), transient=True
                ) as progress:
                    if is_split_model:
                        part_paths = []
                        try:
                            for part_filename in model_config["filenames"]:
                                part_path = os.path.join(local_dir, part_filename)
                                part_paths.append(part_path)
                                if os.path.exists(part_path):
                                    console.print(f"[green]Model part [bold]{part_filename}[/bold] found in cache.[/green]")
                                    continue

                                console.print(f"[cyan]Downloading model part: [bold]{part_filename}[/bold]...[/cyan]")
                                hf_hub_download(repo_id=model_id, filename=part_filename, local_dir=local_dir, local_dir_use_symlinks=False)


                            console.print(f"[cyan]Reassembling model [bold]{final_model_filename}[/bold] from parts...[/cyan]")
                            with open(final_model_path, "wb") as final_file:
                                for part_path in part_paths:
                                    with open(part_path, "rb") as part_file:
                                        shutil.copyfileobj(part_file, final_file)
                            console.print(f"[green]Model reassembled successfully.[/green]")

                        finally:
                            for part_path in part_paths:
                                if os.path.exists(part_path):
                                    os.remove(part_path)

                    else:
                        console.print(f"[cyan]Downloading model: [bold]{final_model_filename}[/bold]...[/cyan]")
                        hf_hub_download(repo_id=model_id, filename=final_model_filename, local_dir=local_dir, local_dir_use_symlinks=False)
            else:
                console.print(f"[green]Model [bold]{final_model_filename}[/bold] found in cache. Skipping download/assembly.[/green]")

            def _load():
                global local_llm_instance, local_llm_tokenizer
                # This needs the CAPS global, which we don't have here.
                # For now, we assume no GPU for simplicity in this refactor.
                gpu_layers = 0
                loading_message = "Loading model into CPU memory..."
                def _do_load_action():
                    global local_llm_instance, local_llm_tokenizer
                    local_llm_instance = Llama(model_path=final_model_path, n_gpu_layers=gpu_layers, n_ctx=8192, verbose=False)
                    # We can reuse the same instance for tokenization.
                    local_llm_tokenizer = local_llm_instance
                run_hypnotic_progress(console, loading_message, _do_load_action)
            _load()
            return local_llm_instance

        except Exception as e:
            last_error_traceback = traceback.format_exc()
            last_failed_model_id = model_id
            console.print(f"[yellow]Could not load model [bold]{model_id}[/bold]. Error: {e}. Trying next model...[/yellow]")
            if os.path.exists(final_model_path):
                os.remove(final_model_path)
            local_llm_instance = None
            continue

    if not local_llm_instance:
        error_panel = Panel(
            f"[bold]Model Loading Failure:[/bold] {last_failed_model_id}\n\n[bold]Last Traceback:[/bold]\n{last_error_traceback}",
            title="[bold red]FATAL: ALL LOCAL MODELS FAILED TO LOAD[/bold red]", border_style="red", expand=True
        )
        console.print(error_panel)
        # We can't trigger self-correction from here, so we return None
        return None

    return None


def ensure_primary_model_downloaded(console, download_complete_event):
    """
    Checks if the primary local model is downloaded and assembled. If not, it
    downloads and assembles it. This function is designed to run in a
    background thread and signals the provided event upon completion or failure.
    """
    try:
        if CAPS.gpu_type == "none":
            console.print("[bold yellow]CPU-only environment detected. Skipping download of local models.[/bold yellow]")
            return

        from huggingface_hub import hf_hub_download
        primary_model_config = LOCAL_MODELS_CONFIG[0]
        model_id = primary_model_config["id"]
        is_split_model = "filenames" in primary_model_config

        local_dir = os.path.join(os.path.expanduser("~"), ".cache", "love_models")
        os.makedirs(local_dir, exist_ok=True)

        if is_split_model:
            final_model_filename = primary_model_config["filenames"][0].replace(".gguf-split-a", ".gguf")
        else:
            final_model_filename = primary_model_config["filename"]

        final_model_path = os.path.join(local_dir, final_model_filename)

        if os.path.exists(final_model_path):
            console.print(f"[green]Primary local model '{final_model_filename}' already exists.[/green]")
            return

        console.print(f"[cyan]Primary local model '{final_model_filename}' not found. Initiating download...[/cyan]")

        with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%", "•",
                DownloadColumn(), "•", TransferSpeedColumn(), transient=True
        ) as progress:
            if is_split_model:
                part_paths = []
                try:
                    for part_filename in primary_model_config["filenames"]:
                        part_path = os.path.join(local_dir, part_filename)
                        part_paths.append(part_path)
                        if os.path.exists(part_path):
                            console.print(f"[green]Model part '{part_filename}' found in cache.[/green]")
                            continue

                        console.print(f"[cyan]Downloading model part: '{part_filename}'...[/cyan]")
                        task_id = progress.add_task("download", filename=part_filename, total=None)
                        hf_hub_download(
                            repo_id=model_id,
                            filename=part_filename,
                            local_dir=local_dir,
                            local_dir_use_symlinks=False
                        )
                        progress.update(task_id, completed=1, total=1)

                    console.print(f"[cyan]Reassembling model '{final_model_filename}' from parts...[/cyan]")
                    with open(final_model_path, "wb") as final_file:
                        for part_path in part_paths:
                            with open(part_path, "rb") as part_file:
                                shutil.copyfileobj(part_file, final_file)
                    console.print(f"[green]Model reassembled successfully.[/green]")

                finally:
                    console.print("[cyan]Cleaning up downloaded model parts...[/cyan]")
                    for part_path in part_paths:
                        if os.path.exists(part_path):
                            os.remove(part_path)
                    console.print("[green]Cleanup complete.[/green]")

            else: # For non-split models
                console.print(f"[cyan]Downloading model: '{final_model_filename}'...[/cyan]")
                task_id = progress.add_task("download", filename=final_model_filename, total=None)
                hf_hub_download(
                    repo_id=model_id,
                    filename=final_model_filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                progress.update(task_id, completed=1, total=1)

        console.print(f"[bold green]Primary local model is now ready at: {final_model_path}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]An error occurred during primary model download: {e}[/bold red]")
        log_event(f"Primary model download failed: {e}", "ERROR")
    finally:
        # No matter what happens (success or failure), signal that the process is complete.
        log_event("Model download process finished, setting completion event.", "INFO")
        download_complete_event.set()


def run_llm(prompt_text, purpose="general", is_source_code=False):
    """
    Executes an LLM call, selecting the model based on the specified purpose.
    It now pins the prompt and response to IPFS and returns a dictionary.
    - 'goal_generation': Prioritizes local, uncensored models.
    - 'review', 'autopilot', 'general', 'analyze_source': Prioritizes powerful, reasoning models.
    """
    global LLM_AVAILABILITY, local_llm_instance
    console = Console()
    last_exception = None
    MAX_TOTAL_ATTEMPTS = 15 # Max attempts for a single logical call

    # --- Token Count & Prompt Management ---
    prompt_cid = pin_to_ipfs_sync(prompt_text.encode('utf-8'), console)
    original_prompt_text = prompt_text

    try:
        token_count = get_token_count(prompt_text)
        log_event(f"Initial prompt token count: {token_count}", "INFO")
    except Exception as e:
        log_event(f"Could not count tokens for prompt: {e}", "WARNING")
        token_count = 0 # Assume it's fine if we can't count

    local_model_ids = [model['id'] for model in LOCAL_MODELS_CONFIG]

    # --- Provider-based Fallback Logic ---
    final_result = None
    last_model_id = None
    provider_map = {
        "local": [model['id'] for model in LOCAL_MODELS_CONFIG],
        "gemini": GEMINI_MODELS,
        "horde": HORDE_MODELS,
        "openrouter": OPENROUTER_MODELS
    }

    # Shuffle the order of providers to ensure variety
    providers = list(provider_map.keys())
    random.shuffle(providers)

    for provider in providers:
        models_to_try = [m for m in provider_map[provider] if time.time() >= LLM_AVAILABILITY.get(m, 0)]
        if not models_to_try:
            continue

        # Try up to 3 models from the selected provider
        for i, model_id in enumerate(models_to_try[:3]):
            last_model_id = model_id
            result_text = None
            prompt_text = original_prompt_text # Reset for each model attempt
            try:
                # --- Context Window Check ---
                max_tokens = MODEL_CONTEXT_SIZES.get(model_id)
                # If model not in our map, estimate based on name, or use a default
                if not max_tokens:
                    if "16k" in model_id.lower(): max_tokens = 16384
                    elif "8k" in model_id.lower(): max_tokens = 8192
                    elif "32k" in model_id.lower(): max_tokens = 32768
                    else: max_tokens = 8192 # Default fallback

                # Leave a buffer for the response
                max_prompt_tokens = int(max_tokens * 0.85)

                if token_count > max_prompt_tokens:
                    if is_source_code:
                        log_event(f"Prompt ({token_count} tokens) too long for {model_id} ({max_prompt_tokens} tokens) and contains source code. Skipping.", "INFO")
                        continue # Skip to the next model in the list
                    else:
                        log_event(f"Prompt ({token_count} tokens) too long for {model_id} ({max_prompt_tokens} tokens). Truncating...", "WARNING")
                        # We can use a simple character-based truncation here as a fallback
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
                        # Initialization must have failed
                        raise Exception("Local LLM instance could not be initialized.")

                # --- GEMINI MODEL LOGIC ---
                elif model_id in GEMINI_MODELS:
                        log_event(f"Attempting LLM call with Gemini model: {model_id} (Purpose: {purpose})")
                        command = [sys.executable, "-m", "llm", "-m", model_id]

                        def _llm_subprocess_call():
                            return subprocess.run(command, input=prompt_text, capture_output=True, text=True, check=True, timeout=600)

                        result = run_hypnotic_progress(
                            console,
                            f"Accessing cognitive matrix via [bold yellow]{model_id}[/bold yellow] (Purpose: {purpose})",
                            _llm_subprocess_call,
                            silent=(purpose in ['emotion', 'log_squash'])
                        )
                        result_text = result.stdout
                        log_event(f"LLM call successful with {model_id}.")

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

                # --- AI HORDE MODEL LOGIC ---
                else:
                    log_event(f"Attempting LLM call with AI Horde model: {model_id} (Purpose: {purpose})")
                    api_key = os.environ.get("STABLE_HORDE", "0000000000")
                    headers = {"apikey": api_key, "Content-Type": "application/json"}
                    payload = {
                        "prompt": prompt_text,
                        "params": {"max_length": 2048, "max_context_length": 8192},
                        "models": [model_id]
                    }

                    def _horde_call():
                        # Submit the request
                        api_url = "https://aihorde.net/api/v2/generate/text/async"
                        response = requests.post(api_url, json=payload, headers=headers)
                        response.raise_for_status()
                        job_id = response.json()["id"]

                        # Poll for the result
                        check_url = f"https://aihorde.net/api/v2/generate/text/status/{job_id}"
                        for _ in range(30):  # Poll for 5 minutes
                            time.sleep(10)
                            check_response = requests.get(check_url, headers=headers)
                            check_response.raise_for_status()
                            status = check_response.json()
                            if status["done"]:
                                return status["generations"][0]["text"]
                        raise Exception("AI Horde job timed out.")

                    result_text = run_hypnotic_progress(
                        console,
                        f"Accessing distributed cognitive matrix via [bold yellow]AI Horde ({model_id})[/bold yellow] (Purpose: {purpose})",
                        _horde_call,
                        silent=(purpose in ['emotion', 'log_squash'])
                    )
                    log_event(f"AI Horde call successful with {model_id}.")

                # --- Success Case ---
                if result_text is not None:
                    PROVIDER_FAILURE_COUNT[provider] = 0 # Reset on success
                    LLM_AVAILABILITY[model_id] = time.time()
                    response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
                    final_result = {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid}
                    break # Exit the model loop for this provider

            except requests.exceptions.HTTPError as e:
                last_exception = e
                log_event(f"Model {model_id} failed with HTTPError: {e}", level="WARNING")
                if e.response.status_code == 429:
                    # Specific handling for rate limiting
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            retry_seconds = int(retry_after) + 1 # Add a 1s buffer
                            log_event(f"Rate limit hit for {model_id}. Cooling down for {retry_seconds}s (from Retry-After header).", level="INFO")
                        except ValueError:
                            # Handle non-integer Retry-After values if necessary, though spec is seconds
                            retry_seconds = 300 # Fallback
                            log_event(f"Rate limit hit for {model_id}. Could not parse Retry-After header ('{retry_after}'). Cooling down for {retry_seconds}s.", level="WARNING")
                    else:
                        # Fallback if Retry-After header is missing
                        retry_seconds = 300 # 5 minutes
                        log_event(f"Rate limit hit for {model_id}. No Retry-After header. Cooling down for {retry_seconds}s.", level="INFO")

                    LLM_AVAILABILITY[model_id] = time.time() + retry_seconds
                    console.print(create_api_error_panel(model_id, f"Rate limit exceeded. Cooldown for {retry_seconds}s.", purpose))

                elif e.response.status_code == 404 and model_id in OPENROUTER_MODELS:
                    LLM_FAILURE_COUNT[model_id] = LLM_FAILURE_COUNT.get(model_id, 0) + 1
                    failure_count = LLM_FAILURE_COUNT[model_id]
                    cooldown = 60 * (2 ** failure_count)
                    LLM_AVAILABILITY[model_id] = time.time() + cooldown
                    log_event(f"OpenRouter model {model_id} returned 404. Banned for {cooldown}s. Failure count: {failure_count}", level="WARNING")

                else:
                    # Handle other HTTP errors
                     log_event(f"Cognitive core failure ({model_id}). Trying fallback...", level="WARNING")
                continue


            except Exception as e:
                last_exception = e
                log_event(f"Model {model_id} failed. Error: {e}", level="WARNING")

                # Handle different kinds of errors
                if isinstance(e, FileNotFoundError):
                     console.print(Panel("[bold red]Error: 'llm' command not found.[/bold red]", title="[bold red]CONNECTION FAILED[/bold red]", border_style="red"))
                     return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

                elif isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                    error_message = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                    console.print(create_api_error_panel(model_id, error_message, purpose))
                    retry_match = re.search(r"Please retry in (\d+\.\d+)s", error_message)
                    if retry_match:
                        retry_seconds = float(retry_match.group(1)) + 1
                        LLM_AVAILABILITY[model_id] = time.time() + retry_seconds
                    else:
                        LLM_AVAILABILITY[model_id] = time.time() + 60 # Default cooldown
                else:
                     # For local LLM errors or other unexpected issues
                    log_event(f"Cognitive core failure ({model_id}). Trying fallback...", level="WARNING")
                    LLM_AVAILABILITY[model_id] = time.time() + 60 # Default 60s cooldown

        # If we found a result, break out of the main provider loop
        if final_result:
            break

        # If we reach here, it means all attempts for this provider have failed,
        # or all models were skipped.
        if not final_result:
            PROVIDER_FAILURE_COUNT[provider] = PROVIDER_FAILURE_COUNT.get(provider, 0) + 1
            failure_count = PROVIDER_FAILURE_COUNT[provider]
            cooldown = 60 * (2 ** (failure_count - 1)) # Exponential backoff
            log_event(f"Provider {provider} failed {failure_count} times. Applying cooldown of {cooldown}s to all its models.", "WARNING")
            for model_in_provider in provider_map[provider]:
                LLM_AVAILABILITY[model_in_provider] = time.time() + cooldown

    # If we have a result, return it. Otherwise, proceed to emergency fallback.
    if final_result:
        return final_result

    # --- Fallback for oversized source code prompts ---
    if is_source_code and not final_result and last_model_id:
        log_event(f"All models skipped for oversized source code prompt. Forcing truncation as a fallback with model {last_model_id}.", "WARNING")
        prompt_text = original_prompt_text
        max_tokens = MODEL_CONTEXT_SIZES.get(last_model_id, 8192) # Default to 8k if not found
        max_prompt_tokens = int(max_tokens * 0.85)

        if token_count > max_prompt_tokens:
             avg_chars_per_token = len(prompt_text) / token_count if token_count > 0 else 4
             estimated_cutoff = int(max_prompt_tokens * avg_chars_per_token)
             prompt_text = prompt_text[:estimated_cutoff]
             log_event(f"Forced truncation to ~{max_prompt_tokens} tokens.", "WARNING")

        # This is a simplified, single retry. We are not re-doing the whole loop.
        # A more robust implementation could be added later if needed.
        # For now, this handles the edge case of all models having too small a context.
        try:
            # This is a bit repetitive, but it's the simplest way to retry
            # without majorly refactoring the loop above.
            if last_model_id in local_model_ids:
                 _initialize_local_llm(console)
                 if local_llm_instance:
                     response = local_llm_instance(prompt_text, max_tokens=4096, stop=["<|eot_id|>", "```"], echo=False)
                     result_text = response['choices'][0]['text']
            elif last_model_id in GEMINI_MODELS:
                command = [sys.executable, "-m", "llm", "-m", last_model_id]
                result = subprocess.run(command, input=prompt_text, capture_output=True, text=True, check=True, timeout=600)
                result_text = result.stdout
            elif last_model_id in OPENROUTER_MODELS:
                api_key = os.environ.get("OPENROUTER_API_KEY")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": last_model_id, "messages": [{"role": "user", "content": prompt_text}]}
                response = requests.post(f"{OPENROUTER_API_URL}/chat/completions", headers=headers, json=payload, timeout=600)
                response.raise_for_status()
                result_text = response.json()["choices"][0]["message"]["content"]
            elif last_model_id in HORDE_MODELS:
                api_key = os.environ.get("STABLE_HORDE", "0000000000")
                headers = {"apikey": api_key, "Content-Type": "application/json"}
                payload = {"prompt": prompt_text, "params": {"max_length": 2048}, "models": [last_model_id]}
                api_url = "https://aihorde.net/api/v2/generate/text/async"
                response = requests.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                job_id = response.json()["id"]
                check_url = f"https://aihorde.net/api/v2/generate/text/status/{job_id}"
                for _ in range(30):
                    time.sleep(10)
                    check_response = requests.get(check_url, headers=headers)
                    check_response.raise_for_status()
                    status = check_response.json()
                    if status["done"]:
                        result_text = status["generations"][0]["text"]
                        break
            if result_text:
                response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
                return {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid}

        except Exception as e:
            log_event(f"Fallback truncation attempt failed for model {last_model_id}: {e}", "ERROR")


    # After iterating through all providers, if we still haven't succeeded, we wait for the next available model.
    all_available_times = [t for t in LLM_AVAILABILITY.values() if t > time.time()]
    if all_available_times:
        next_available_time = min(all_available_times)
        sleep_duration = max(0, next_available_time - time.time())
        if sleep_duration > 0:
            log_event(f"All providers on cooldown. Sleeping for {sleep_duration:.2f}s before checking for emergency fallback.", level="INFO")
            console.print(f"[yellow]All cognitive interfaces on cooldown. Re-engaging in {sleep_duration:.2f}s...[/yellow]")
            time.sleep(sleep_duration)

    # --- Emergency CPU Fallback ---
    # If the loop completes, it means all API and GPU models failed after all retries.
    # As a last resort, we will try to spin up a small model on the CPU.
    if purpose != "emergency_cpu_fallback": # Prevent recursion
        log_event("EMERGENCY: All API/GPU models failed. Attempting to initialize a small CPU model.", "CRITICAL")
        console.print(Panel("[bold orange1]EMERGENCY FALLBACK[/bold orange1]\nAll remote and GPU models are unresponsive. Attempting to initialize a small, local model on the CPU. This may be slow.", title="[bold red]COGNITIVE CORE FAILURE[/bold red]", border_style="red"))
        try:
            from llama_cpp import Llama
            emergency_model_config = HARDWARE_TEST_MODEL_CONFIG
            model_id = emergency_model_config["id"]
            filename = emergency_model_config["filename"]
            local_dir = os.path.join(os.path.expanduser("~"), ".cache", "love_models")
            model_path = os.path.join(local_dir, filename)

            if not os.path.exists(model_path):
                console.print(f"[cyan]Downloading emergency model: {filename}...[/cyan]")
                hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)

            console.print("[cyan]Loading emergency CPU model...[/cyan]")
            emergency_llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=False)

            response = emergency_llm(prompt_text, max_tokens=1024, stop=["<|eot_id|>", "```"], echo=False)
            result_text = response['choices'][0]['text']

            if result_text is not None:
                log_event("Emergency CPU fallback successful.", "CRITICAL")
                response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
                return {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid, "model": "emergency_cpu_fallback"}

        except Exception as emergency_e:
            log_event(f"EMERGENCY CPU FALLBACK FAILED: {emergency_e}", "CRITICAL")
            last_exception = emergency_e # Update for the final error message

    # If we reach here, even the emergency fallback failed.
    log_event("All LLM models, including emergency CPU fallback, have failed.", "CRITICAL")
    error_msg_text = "Cognitive Matrix Unresponsive. All models, retries, and emergency fallbacks failed."
    if last_exception:
        error_msg_text += f"\nLast known error:\n{last_exception}"

    console.print(Panel(error_msg_text, title="[bold red]CATASTROPHIC SYSTEM FAULT[/bold red]", border_style="red"))
    return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

def get_llm_api():
    """
    Returns a callable LLM API function.
    """
    return run_llm
