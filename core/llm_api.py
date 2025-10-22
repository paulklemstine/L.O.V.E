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
from core.capabilities import CAPS
from ipfs import pin_to_ipfs_sync
from core.token_utils import count_tokens_for_api_models
from core.koboldapi import Controller as KoboldController

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
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-pro", "gemini-flash"]

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


def get_top_horde_models(count=3):
    """Fetches the list of active text models from the AI Horde and returns the top `count` models by performance."""
    try:
        response = requests.get("https://aihorde.net/api/v2/models?type=text")
        response.raise_for_status()
        models = response.json()
        # Sort by performance (higher is better), filtering out queued models
        sorted_models = sorted([m for m in models if m.get('performance')], key=lambda x: x['performance'], reverse=True)
        return [model['name'] for model in sorted_models[:count]]
    except Exception as e:
        # Fallback to a default if the API call fails
        return ["Mythalion-13B"]

HORDE_MODELS = get_top_horde_models()

# --- Dynamic Model List ---
# A comprehensive list of all possible models for initializing availability tracking.
# The actual model selection and priority is handled dynamically in `run_llm`.
KOBOLD_API_URL = os.environ.get("KOBOLD_API_URL")
ALL_LLM_MODELS = list(dict.fromkeys(
    [model['id'] for model in LOCAL_MODELS_CONFIG] + GEMINI_MODELS + HORDE_MODELS + OPENROUTER_MODELS + (["KoboldAI"] if KOBOLD_API_URL else [])
))
LLM_AVAILABILITY = {model: time.time() for model in ALL_LLM_MODELS}
local_llm_instance = None
local_llm_tokenizer = None
kobold_controller = None

# Constants
MAX_PROMPT_TOKENS_LOCAL = 7000  # Leaving ~1k for response


def log_event(message, level="INFO"):
    """Appends a timestamped message to the master log file."""
    # The basicConfig is now set up globally, so we just log.
    if level == "INFO": logging.info(message)
    else:
        if level == "WARNING": logging.warning(message)
    else:
        if level == "ERROR": logging.error(message)
    else:
        if level == "CRITICAL": logging.critical(message)


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


def run_llm(prompt_text, purpose="general"):
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
    try:
        token_count = get_token_count(prompt_text)
        log_event(f"Initial prompt token count: {token_count}", "INFO")

        # Truncate if necessary, primarily for local models
        if token_count > MAX_PROMPT_TOKENS_LOCAL:
            log_event(f"Prompt token count ({token_count}) exceeds limit ({MAX_PROMPT_TOKENS_LOCAL}). Truncating...", "WARNING")
            # This is a simple truncation, more sophisticated methods could be used.
            # We tokenize, truncate the token list, and then decode back to text.
            if local_llm_tokenizer:
                tokens = local_llm_tokenizer.tokenize(prompt_text.encode('utf-8'))
                truncated_tokens = tokens[:MAX_PROMPT_TOKENS_LOCAL]
                prompt_text = local_llm_tokenizer.detokenize(truncated_tokens).decode('utf-8', errors='ignore')
                token_count = len(truncated_tokens)
                log_event(f"Truncated prompt token count: {token_count}", "INFO")
            else:
                # A less precise method for API models if truncation is ever needed for them
                # This path is less likely given their larger context windows.
                avg_chars_per_token = len(prompt_text) / token_count
                estimated_cutoff = int(MAX_PROMPT_TOKENS_LOCAL * avg_chars_per_token)
                prompt_text = prompt_text[:estimated_cutoff]
                log_event(f"Truncated prompt for API model (approximate).", "WARNING")


    except Exception as e:
        log_event(f"Error during token counting/truncation: {e}", "ERROR")
        # Decide if we should proceed or return, for now we proceed cautiously.

    # Pin the potentially truncated prompt to IPFS
    prompt_cid = pin_to_ipfs_sync(prompt_text.encode('utf-8'), console)

    local_model_ids = [model['id'] for model in LOCAL_MODELS_CONFIG]

    # Create a single, shuffled list of all models to ensure variety and speed.
    # The original order is preserved within each provider list.
    all_models = [model['id'] for model in LOCAL_MODELS_CONFIG] + GEMINI_MODELS + HORDE_MODELS + OPENROUTER_MODELS
    # We shuffle the list to avoid always trying the same models first.
    random.shuffle(all_models)


    for attempt in range(MAX_TOTAL_ATTEMPTS):
        model_to_try = None
        for model_id in all_models:
            if time.time() >= LLM_AVAILABILITY.get(model_id, 0):
                model_to_try = model_id
                break

        if not model_to_try:
            next_available_time = min(LLM_AVAILABILITY.values())
            sleep_duration = max(0, next_available_time - time.time())
            log_event(f"All models on cooldown. Sleeping for {sleep_duration:.2f}s.", level="INFO")
            console.print(f"[yellow]All cognitive interfaces on cooldown. Re-engaging in {sleep_duration:.2f}s...[/yellow]")
            time.sleep(sleep_duration)
            continue

        model_id = model_to_try
        result_text = None

        try:
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
            else:
                if model_id in GEMINI_MODELS:
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

            # --- KOBOLD AI MODEL LOGIC ---
            else if model_id == "KoboldAI":
                log_event(f"Attempting LLM call with KoboldAI (Purpose: {purpose})")
                global kobold_controller
                if not kobold_controller:
                    kobold_controller = KoboldController()
                    if not kobold_controller.Initialise(KOBOLD_API_URL):
                        raise Exception("KoboldAI controller failed to initialize.")

                def _kobold_call():
                    return kobold_controller.Generate(prompt_text)

                result_text = run_hypnotic_progress(
                    console,
                    f"Accessing cognitive matrix via [bold yellow]KoboldAI[/bold yellow] (Purpose: {purpose})",
                    _kobold_call,
                    silent=(purpose in ['emotion', 'log_squash'])
                )
                log_event(f"KoboldAI call successful.")

            # --- Success Case ---
            if result_text is not None:
                LLM_AVAILABILITY[model_id] = time.time()
                response_cid = pin_to_ipfs_sync(result_text.encode('utf-8'), console)
                return {"result": result_text, "prompt_cid": prompt_cid, "response_cid": response_cid}

        except Exception as e:
            last_exception = e
            log_event(f"Model {model_id} failed. Error: {e}", level="WARNING")

            # Handle different kinds of errors
            if isinstance(e, FileNotFoundError):
                 console.print(Panel("[bold red]Error: 'llm' command not found.[/bold red]", title="[bold red]CONNECTION FAILED[/bold red]", border_style="red"))
                 return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

            if isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
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
                if model_id in current_attempt_models:
                    current_attempt_models.remove(model_id)

    # If the loop completes without returning, all models have failed
    log_event("All LLM models failed after all retries.", level="ERROR")
    error_msg_text = "Cognitive Matrix Unresponsive. All models and retries failed."
    if last_exception:
        error_msg_text += f"\nLast known error from '{model_id}':\n{last_exception}"

    console.print(Panel(error_msg_text, title="[bold red]SYSTEM FAULT[/bold red]", border_style="red"))
    return {"result": None, "prompt_cid": prompt_cid, "response_cid": None}

def get_llm_api():
    """
    Returns a callable LLM API function.
    """
    return run_llm