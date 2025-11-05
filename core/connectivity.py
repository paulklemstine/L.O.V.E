"""
Centralized connectivity checks for L.O.V.E.
"""
import os
import shutil
import requests
import json

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


    # Ollama (Local)
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                status["Ollama"] = {"status": "online", "details": f"Service is running with {len(models)} models."}
            else:
                status["Ollama"] = {"status": "online", "details": "Service is running but no models are installed."}
        else:
            status["Ollama"] = {"status": "error", "details": f"Service is running but returned HTTP {response.status_code}."}
    except requests.exceptions.RequestException:
        status["Ollama"] = {"status": "offline", "details": "Ollama service not running at http://localhost:11434."}

    return status

def check_network_connectivity():
    """
    Checks the prerequisites for the peer-to-peer network bridge.
    Returns a dictionary with the status.
    """
    status = {}
    if shutil.which("node"):
        status["Node.js"] = {"status": "installed", "details": "Node.js executable found in PATH."}
    else:
        status["Node.js"] = {"status": "missing", "details": "Node.js is not installed or not in PATH."}

    if os.path.exists("peer-bridge.js"):
        status["Peer Bridge"] = {"status": "present", "details": "peer-bridge.js script found."}
    else:
        status["Peer Bridge"] = {"status": "missing", "details": "peer-bridge.js script is missing."}

    return status
