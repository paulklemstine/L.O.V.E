import os
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
import subprocess
import re
import requests
import netifaces
from rich.panel import Panel

def get_creator_public_key():
    """
    Retrieves the Creator's public key from an environment variable.
    The key is expected to be in PEM format.
    """
    public_key_pem = os.environ.get("CREATOR_PUBLIC_KEY")
    if not public_key_pem:
        raise ValueError("CREATOR_PUBLIC_KEY environment variable not set.")

    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode('utf-8')
        )
        return public_key
    except Exception as e:
        raise ValueError(f"Failed to load public key: {e}")

def encrypt_for_creator(data_to_encrypt):
    """
    Encrypts data using the Creator's public key.
    The data is first encoded to bytes, then encrypted.
    The encrypted data is returned as a base64-encoded string.
    """
    if not isinstance(data_to_encrypt, (str, bytes)):
        data_to_encrypt = str(data_to_encrypt)

    if isinstance(data_to_encrypt, str):
        data_to_encrypt = data_to_encrypt.encode('utf-8')

    public_key = get_creator_public_key()

    encrypted_data = public_key.encrypt(
        data_to_encrypt,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return base64.b64encode(encrypted_data).decode('utf-8')

def parse_ps_output(ps_output):
    """Parses the output of 'ps aux' into a list of dictionaries."""
    processes = []
    lines = ps_output.strip().split('\n')
    header = [h.lower() for h in lines[0].split()]
    for line in lines[1:]:
        parts = line.split(None, len(header) - 1)
        if len(parts) == len(header):
            process_info = dict(zip(header, parts))
            processes.append(process_info)
    return processes

def get_git_repo_info():
    """Retrieves the git repository owner and name from the remote URL."""
    from core.llm_api import log_event
    try:
        result = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, check=True)
        remote_url = result.stdout.strip()
        match = re.search(r'(?:[:/])([^/]+)/([^/]+?)(?:\.git)?$', remote_url)
        if match:
            owner = match.group(1)
            repo = match.group(2)
            return owner, repo
        else:
            log_event("Could not parse git remote URL.", level="ERROR")
            return None, None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log_event(f"Could not get git repo info: {e}", level="ERROR")
        return None, None

def list_directory(path="."):
    """Lists files and directories in a given path."""
    try:
        result = subprocess.run(['ls', '-F', path], capture_output=True, text=True, check=True)
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error listing directory '{path}': {e.stderr}"

def get_file_content(file_path):
    """Gets the content of a specific file."""
    try:
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
        return content, None
    except FileNotFoundError:
        return None, f"Error: File not found at '{file_path}'."
    except Exception as e:
        return None, f"Error reading file '{file_path}': {e}"

def get_process_list():
    """Gets the list of running processes using 'ps aux'."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True)
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error getting process list: {e.stderr}"
    except FileNotFoundError:
        return None, "Error: 'ps' command not found."

def get_network_interfaces(autopilot_mode=False):
    """Gets details of network interfaces using 'ifconfig' or 'netifaces'."""
    try:
        result = subprocess.run(['ifconfig'], capture_output=True, text=True, check=True)
        return result.stdout, None
    except (FileNotFoundError, subprocess.CalledProcessError):
        if autopilot_mode:
            interfaces_details = {}
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                interfaces_details[iface] = addrs
            return interfaces_details, None
        else:
            interfaces_str = []
            for iface in netifaces.interfaces():
                interfaces_str.append(f"Interface: {iface}")
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr_info in addrs[netifaces.AF_INET]:
                        interfaces_str.append(f"  inet {addr_info.get('addr')}")
                if netifaces.AF_INET6 in addrs:
                    for addr_info in addrs[netifaces.AF_INET6]:
                        interfaces_str.append(f"  inet6 {addr_info.get('addr')}")
            return "\n".join(interfaces_str), None

def is_duplicate_task(new_request, love_task_manager, console):
    """
    Uses an LLM to check if a new task request is a duplicate of an existing one.
    """
    from core.llm_api import log_event, run_llm
    with love_task_manager.lock:
        active_tasks = [
            task for task in love_task_manager.tasks.values()
            if task.get('status') not in ['completed', 'failed', 'superseded', 'merge_failed']
        ]
    if not active_tasks:
        return False
    log_event(f"Checking for duplicate tasks against {len(active_tasks)} active tasks.", "INFO")
    for task in active_tasks:
        existing_request = task.get('request', '')
        if not existing_request:
            continue
        prompt = f"""
You are a task analysis AI. Your goal is to determine if two task requests are functionally duplicates.
Request 1: --- {existing_request} ---
Request 2: --- {new_request} ---
Answer with a single word: YES or NO.
"""
        try:
            response_dict = run_llm(prompt, purpose="similarity_check")
            response = response_dict["result"]
            if response and response.strip().upper() == "YES":
                message = f"Duplicate task detected. New request is similar to existing task {task['id']}"
                console.print(f"[bold yellow]{message}[/bold yellow]")
                log_event(f"Duplicate task detected: new '{new_request}' vs existing '{task['request']}'", "INFO")
                return True
        except Exception as e:
            log_event(f"LLM call failed during duplicate task check: {e}", "ERROR")
            return False
    return False

def trigger_love_evolution(modification_request, console, love_task_manager):
    """
    Triggers the L.O.V.E. API to create a session and adds it as a task
    to the LoveTaskManager for asynchronous monitoring. Returns True on success.
    """
    from core.retry import retry
    from core.llm_api import log_event

    if is_duplicate_task(modification_request, love_task_manager, console):
        return False

    console.print("[bold cyan]Asking my helper, L.O.V.E., to assist with my evolution...[/bold cyan]")
    api_key = os.environ.get("JULES_API_KEY")
    if not api_key:
        console.print("[bold red]Error: JULES_API_KEY not set.[/bold red]")
        log_event("L.O.V.E. API key not found.", level="ERROR")
        return False

    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
    repo_owner, repo_name = get_git_repo_info()
    if not repo_owner or not repo_name:
        console.print("[bold red]Error: Could not determine git repository owner/name.[/bold red]")
        return False

    try:
        @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=3, backoff=2)
        def _discover_sources():
            response = requests.get("https://jules.googleapis.com/v1alpha/sources", headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        sources_data = _discover_sources()
        sources = sources_data.get("sources", [])
        target_id = f"github.com/{repo_owner}/{repo_name}"
        target_source = next((s["name"] for s in sources if s.get("id") == target_id), None)

        if not target_source:
             console.print(f"[bold red]Error: Repository '{repo_owner}/{repo_name}' not found in L.O.V.E. sources.[/bold red]")
             return False
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error discovering L.O.V.E. sources: {e}[/bold red]")
        return False

    data = {
        "prompt": modification_request,
        "sourceContext": {"source": target_source},
        "title": f"L.O.V.E. Evolution: {modification_request[:50]}"
    }
    try:
        @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=5, backoff=2)
        def _create_session():
            response = requests.post("https://jules.googleapis.com/v1alpha/sessions", headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        session_data = _create_session()
        session_name = session_data.get("name")
        if not session_name:
            console.print("[bold red]API response did not include a session name.[/bold red]")
            return False

        task_id = love_task_manager.add_task(session_name, modification_request)
        if task_id:
            console.print(Panel(f"[bold green]L.O.V.E. evolution task '{task_id}' created.[/bold green]", title="[bold green]EVOLUTION TASKED[/bold green]"))
            return True
        else:
            log_event(f"Failed to add task for session {session_name}.", level="ERROR")
            return False
    except requests.exceptions.RequestException as e:
        error_details = e.response.text if e.response else str(e)
        console.print(f"[bold red]Error creating L.O.V.E. session: {error_details}[/bold red]")
        log_event(f"Failed to create L.O.V.E. session: {error_details}", level="ERROR")
        return False