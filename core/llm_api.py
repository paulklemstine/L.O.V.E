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

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
from bbs import run_hypnotic_progress
from huggingface_hub import hf_hub_download
from display import create_api_error_panel
from evolve import CAPS

# --- CONFIGURATION & GLOBALS ---
# A list of local GGUF models to try in sequence. If the first one fails
# (e.g., due to insufficient VRAM), the script will fall back to the next.
HARDWARE_TEST_MODEL_CONFIG = {
    "id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
}

LOCAL_MODELS_CONFIG = [
    {
        "id": "TheBloke/CodeLlama-70B-Instruct-GGUF",
        "filenames": ["codellama-70b-instruct.Q8_0.gguf-split-a","codellama-70b-instruct.Q8_0.gguf-split-b"]

    },
    {
        "id": "bartowski/Llama-3.3-70B-Instruct-ablated-GGUF",
        "filename": "Llama-3.3-70B-Instruct-ablated-IQ4_XS.gguf"
    },
    {
        "id": "bartowski/deepseek-r1-qwen-2.5-32B-ablated-GGUF",
        "filename": "deepseek-r1-qwen-2.5-32B-ablated-IQ4_XS.gguf"
    }
]

# --- Fallback Model Configuration ---
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
HORDE_MODELS = ["koboldcpp/Fimbulvetr-11B-v2", "PygmalionAI/pygmalion-2-70b"]

# --- Dynamic Model List ---
# A comprehensive list of all possible models for initializing availability tracking.
# The actual model selection and priority is handled dynamically in `run_llm`.
ALL_LLM_MODELS = list(dict.fromkeys(
    [model['id'] for model in LOCAL_MODELS_CONFIG] + GEMINI_MODELS + HORDE_MODELS
))
LLM_AVAILABILITY = {model: time.time() for model in ALL_LLM_MODELS}
local_llm_instance = None


def log_event(message, level="INFO"):
    """Appends a timestamped message to the master log file."""
    # The basicConfig is now set up globally, so we just log.
    if level == "INFO": logging.info(message)
    elif level == "WARNING": logging.warning(message)
    elif level == "ERROR": logging.error(message)
    elif level == "CRITICAL": logging.critical(message)


def _initialize_local_llm(console):
    """
    Iterates through the configured local models, attempting to download,
    reassemble (if split), and initialize each one in sequence.
    If all fail, it triggers self-correction.
    """
    global local_llm_instance
    if local_llm_instance:
        return local_llm_instance

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
                global local_llm_instance
                # This needs the CAPS global, which we don't have here.
                # For now, we assume no GPU for simplicity in this refactor.
                gpu_layers = 0
                loading_message = "Loading model into CPU memory..."
                def _do_load_action():
                    global local_llm_instance
                    local_llm_instance = Llama(model_path=final_model_path, n_gpu_layers=gpu_layers, n_ctx=8192, verbose=False)
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


def ensure_primary_model_downloaded(console):
    """
    Checks if the primary local model (the first in the config) is downloaded
    and assembled. If not, it downloads the parts and assembles them.
    This is a blocking function intended for pre-flight checks.
    """
    if CAPS.gpu_type == "none":
        console.print("[bold yellow]CPU-only environment detected. Skipping download of local models.[/bold yellow]")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[bold red]LLM API Error: huggingface_hub not installed.[/bold red]")
        return

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
                        # Progress bar handling is manual with this setup
                    )
                    progress.update(task_id, completed=1, total=1) # Mark as complete for visual feedback

                console.print(f"[cyan]Reassembling model '{final_model_filename}' from parts...[/cyan]")
                with open(final_model_path, "wb") as final_file:
                    for part_path in part_paths:
                        with open(part_path, "rb") as part_file:
                            shutil.copyfileobj(part_file, final_file)
                console.print(f"[green]Model reassembled successfully.[/green]")

            finally:
                # Clean up the split parts after assembly
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


def run_llm(prompt_text, purpose="general", use_premium_horde=False):
    """
    Executes an LLM call, selecting the model based on the specified purpose.
    - 'goal_generation': Prioritizes local, uncensored models.
    - 'review', 'autopilot', 'general', 'analyze_source': Prioritizes powerful, reasoning models.
    - 'use_premium_horde': Forces the use of a large, premium Horde model.
    """
    global LLM_AVAILABILITY, local_llm_instance
    console = Console()
    last_exception = None
    MAX_TOTAL_ATTEMPTS = 15 # Max attempts for a single logical call

    local_model_ids = [model['id'] for model in LOCAL_MODELS_CONFIG]

    # Dynamically set model priority based on purpose
    if use_premium_horde:
        # Force the use of the premium horde model first.
        llm_models_priority = ["PygmalionAI/pygmalion-2-70b"] + GEMINI_MODELS + local_model_ids
        log_event(f"Running LLM with premium Horde model. Priority: Pygmalion-70b -> Gemini -> Local.", level="INFO")
    elif purpose == 'emotion':
        # Prioritize the fastest, cheapest models for non-critical personality updates
        llm_models_priority = sorted(GEMINI_MODELS, key=lambda m: 'flash' not in m) + local_model_ids
        log_event(f"Running LLM for purpose '{purpose}'. Priority: Flash -> Pro -> Local.", level="INFO")
    elif purpose == 'goal_generation':
        # Prioritize local ablated models for creative/unrestricted tasks
        llm_models_priority = local_model_ids + GEMINI_MODELS
        log_event(f"Running LLM for purpose '{purpose}'. Priority: Local -> Gemini.", level="INFO")
    else:  # Covers 'review', 'autopilot', 'general', and 'analyze_source'
        # Prioritize powerful Gemini models for reasoning tasks
        llm_models_priority = GEMINI_MODELS + HORDE_MODELS + local_model_ids
        log_event(f"Running LLM for purpose '{purpose}'. Priority: Gemini -> Horde -> Local.", level="INFO")

    # This list will be mutated if a model fails catastrophically
    current_attempt_models = list(llm_models_priority)

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        available_models = sorted(
            [(model, available_at) for model, available_at in LLM_AVAILABILITY.items() if time.time() >= available_at and model in current_attempt_models],
            key=lambda x: current_attempt_models.index(x[0])
        )

        if not available_models:
            next_available_time = min(LLM_AVAILABILITY.values())
            sleep_duration = max(0, next_available_time - time.time())
            log_event(f"All models on cooldown. Sleeping for {sleep_duration:.2f}s.", level="INFO")
            console.print(f"[yellow]All cognitive interfaces on cooldown. Re-engaging in {sleep_duration:.2f}s...[/yellow]")
            time.sleep(sleep_duration)
            continue

        model_id, _ = available_models[0]

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
                    LLM_AVAILABILITY[model_id] = time.time()
                    return result_text
                else:
                    # Initialization must have failed
                    raise Exception("Local LLM instance could not be initialized.")

            # --- HORDE MODEL LOGIC ---
            elif model_id in HORDE_MODELS:
                log_event(f"Attempting to use Kobold AI Horde model: {model_id}")
                api_key = os.environ.get("STABLE_HORDE")
                if not api_key:
                    raise Exception("STABLE_HORDE environment variable not set.")

                headers = {"apikey": api_key, "Content-Type": "application/json"}
                payload = {
                    "prompt": prompt_text,
                    "params": {"max_length": 4096},
                    "models": [model_id]
                }

                def _horde_call():
                    # Start async generation
                    async_url = "https://stablehorde.net/api/v2/generate/text/async"
                    response = requests.post(async_url, json=payload, headers=headers)
                    response.raise_for_status()
                    job_id = response.json().get('id')
                    if not job_id:
                        raise Exception("Failed to get job ID from Horde API.")

                    # Poll for result
                    check_url = f"https://stablehorde.net/api/v2/generate/text/status/{job_id}"
                    for _ in range(120): # Poll for up to 2 minutes
                        time.sleep(2) # Be gentle with the API
                        status_response = requests.get(check_url, headers=headers)
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data.get('done'):
                                return status_data['generations'][0]['text']
                    raise Exception("Horde generation timed out after 2 minutes.")

                result_text = run_hypnotic_progress(
                    console,
                    f"Processing via Kobold AI Horde [bold yellow]{model_id}[/bold yellow] (Purpose: {purpose})",
                    _horde_call,
                    silent=(purpose in ['emotion', 'log_squash'])
                )
                log_event(f"Horde call successful with {model_id}.")
                LLM_AVAILABILITY[model_id] = time.time()
                return result_text

            # --- GEMINI MODEL LOGIC ---
            else:
                log_event(f"Attempting LLM call with Gemini model: {model_id} (Purpose: {purpose})")
                command = ["llm", "-m", model_id]

                def _llm_subprocess_call():
                    return subprocess.run(command, input=prompt_text, capture_output=True, text=True, check=True, timeout=600)

                result = run_hypnotic_progress(
                    console,
                    f"Accessing cognitive matrix via [bold yellow]{model_id}[/bold yellow] (Purpose: {purpose})",
                    _llm_subprocess_call,
                    silent=(purpose in ['emotion', 'log_squash'])
                )
                log_event(f"LLM call successful with {model_id}.")
                LLM_AVAILABILITY[model_id] = time.time()
                return result.stdout

        except Exception as e:
            last_exception = e
            log_event(f"Model {model_id} failed. Error: {e}", level="WARNING")

            # Handle different kinds of errors
            if isinstance(e, FileNotFoundError):
                 console.print(Panel("[bold red]Error: 'llm' command not found.[/bold red]", title="[bold red]CONNECTION FAILED[/bold red]", border_style="red"))
                 return None # Fatal error if llm command is missing

            if isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                error_message = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)

                # Use the new styled panel for API errors.
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
                    # Remove the failing model from this attempt's list
                    current_attempt_models.remove(model_id)

    # If the loop completes without returning, all models have failed
    log_event("All LLM models failed after all retries.", level="ERROR")
    error_msg_text = "Cognitive Matrix Unresponsive. All models and retries failed."
    if last_exception:
        error_msg_text += f"\nLast known error from '{model_id}':\n{last_exception}"

    console.print(Panel(error_msg_text, title="[bold red]SYSTEM FAULT[/bold red]", border_style="red"))
    return None