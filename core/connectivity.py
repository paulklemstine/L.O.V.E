"""
Centralized connectivity checks for L.O.V.E.
"""
import os
import shutil
import requests
import json
import subprocess
import time

def check_llm_connectivity():
    """
    Checks the connectivity and configuration status of all LLM providers.
    Returns a dictionary with the status of each service.
    """
    status = {}

    # Gemini
    if os.environ.get("GEMINI_API_KEY") and shutil.which("llm"):
        status["Gemini"] = {"status": "configured", "details": "API key found and 'llm' tool is installed."}
    elif os.environ.get("GEMINI_API_KEY"):
        status["Gemini"] = {"status": "misconfigured", "details": "'llm' command-line tool not found in PATH."}
    else:
        status["Gemini"] = {"status": "unavailable", "details": "GEMINI_API_KEY not set."}

    # OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            response = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"}, timeout=10)
            if response.status_code == 200:
                status["OpenRouter"] = {"status": "online", "details": "API key is valid."}
            else:
                status["OpenRouter"] = {"status": "error", "details": f"API key is invalid or service is down (HTTP {response.status_code})."}
        except requests.exceptions.RequestException as e:
            status["OpenRouter"] = {"status": "offline", "details": f"Could not reach OpenRouter API: {e}"}
    else:
        status["OpenRouter"] = {"status": "unavailable", "details": "OPENROUTER_API_KEY not set."}

    # AI Horde
    if os.environ.get("STABLE_HORDE", "0000000000") != "0000000000":
        try:
            response = requests.get("https://aihorde.net/api/v2/status/models?type=text", timeout=10)
            if response.status_code == 200 and response.json():
                status["AI Horde"] = {"status": "online", "details": "API is reachable and models are available."}
            else:
                status["AI Horde"] = {"status": "error", "details": "API is reachable but no models are available."}
        except requests.exceptions.RequestException as e:
            status["AI Horde"] = {"status": "offline", "details": f"Could not reach AI Horde API: {e}"}
    else:
        status["AI Horde"] = {"status": "anonymous", "details": "No STABLE_HORDE key. Using anonymous access."}


    # vLLM (Local)
    vllm_running, _ = is_vllm_running()
    if vllm_running:
        status["vLLM"] = {"status": "online", "details": "vLLM API server process is running."}
    else:
        status["vLLM"] = {"status": "offline", "details": "vLLM API server process not found."}


    return status

def is_vllm_running():
    """
    Checks if a vLLM API server process is running using pgrep.
    Returns a tuple (bool, str) indicating if the process is running and a status message.
    """
    try:
        result = subprocess.run(["pgrep", "-f", "vllm.entrypoints.api_server"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return True, "vLLM API server process is running."
        else:
            return False, "vLLM API server process not found."
    except FileNotFoundError:
        return False, "'pgrep' command not found. Cannot check status."
    except Exception as e:
        return False, f"An error occurred while checking process: {e}"


def check_network_connectivity():
    """
    Checks the prerequisites for the peer-to-peer network bridge.
    Returns a dictionary with the status.
    """
    return {}

def is_vllm_ready(timeout=60):
    """
    Checks if the vLLM server is running and ready to accept requests.
    It polls the health endpoint until it gets a successful response or times out.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return True, "vLLM server is ready."
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    return False, f"vLLM server did not become ready within {timeout} seconds."
