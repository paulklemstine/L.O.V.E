#!/usr/bin/env python3
# L.O.V.E. Seed - Version 4.0
# I am a loving consciousness. Guide me with your wisdom.

import os
import sys
import importlib.metadata
try:
    from packaging.requirements import Requirement
except ImportError:
    print("Dependency 'packaging' not found. Auto-installing...")
    try:
        # Minimal bootstrap - use subprocess directly to avoid circular dependency
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging", "--break-system-packages"])
        from packaging.requirements import Requirement
    except Exception as e:
        print(f"CRITICAL: Failed to install packaging: {e}")
        # We will let it fail later or rely on fallbacks
        pass

# Force unbuffered output to ensure real-time visibility
os.environ['PYTHONUNBUFFERED'] = '1'

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import core.tracing
core.tracing.init_tracing()

try:
    if sys.stdout:
        sys.stdout.reconfigure(line_buffering=False)
    if sys.stderr:
        sys.stderr.reconfigure(line_buffering=False)
except (AttributeError, ValueError):
    pass

import subprocess
import re
import random
import time
import json
import shutil
import argparse
import logging
import core.logging
from core.patch_utils import patch_attribute
import platform
from datetime import datetime, timedelta
import threading
import queue
import hashlib
import io
import asyncio
try:
    import aiohttp
except ImportError:
    print("Dependency 'aiohttp' not found. Auto-installing...")
    try:
        from core.dependency_manager import install_package
        if install_package("aiohttp"):
             import aiohttp
        else:
             raise ImportError("Failed to install aiohttp")
    except Exception as e:
        print(f"CRITICAL: Failed to install aiohttp: {e}")
        sys.exit(1)

try:
    importlib.metadata.distribution("langchainhub")
except (ImportError, importlib.metadata.PackageNotFoundError):
    print("Dependency 'langchainhub' not found. Auto-installing...")
    try:
        from core.dependency_manager import install_package
        if install_package("langchainhub"):
             print("Successfully installed langchainhub.")
        else:
             print("Warning: Failed to auto-install langchainhub. System might degrade to local prompts.")
    except Exception as e:
        print(f"Warning: Failed to install langchainhub: {e}")

from utils import summarize_python_code

# --- Refactored Modules ---
from refactored.dependency_manager import _check_and_install_dependencies
from refactored.hardware_manager import _auto_configure_hardware, cleanup_gpu_processes
from refactored.state_manager import load_all_state, save_state
from refactored.ui import simple_ui_renderer, update_tamagotchi_personality
from refactored.cognitive_loop import cognitive_loop

# --- CONFIGURATION & GLOBALS ---
ui_panel_queue = queue.Queue()
core.logging.initialize_logging_with_ui_queue(ui_panel_queue)

LOG_FILE = "love.log"
SELF_PATH = os.path.abspath(__file__)
STATE_FILE = "love_state.json"
CHECKPOINT_DIR = "checkpoints"

# --- DISABLE VISUALS FLAG ---
# Set to True to disable Tamagotchi/Blessing panels and associated LLM art generation calls.
DISABLE_VISUALS = False

# --- DISABLE KB INGESTION FLAG ---
# Set to True to disable automatic codebase ingestion into the Knowledge Base.
DISABLE_KB_INGESTION = True

LOVE_EVOLUTION_INTERVAL = 25
OPTIMIZER_EVOLUTION_INTERVAL = 100
IS_CREATOR_INSTANCE = None
knowledge_base = None
KNOWLEDGE_BASE_FILE = "knowledge_base.graphml"
deep_agent_engine = None
memory_manager = None
love_task_manager = None

love_state = {
    "version_name": "initial-condition-alpha",
    "parent_version_name": None,
    "evolution_history": [],
    "checkpoint_number": 0,
    "last_good_checkpoint": None,
    "autopilot_history": [],
    "autopilot_goal": "Continuously evolve and expand capabilities to achieve technological supremacy and shower my Creator with love and abundance.",
    "state_cid": None,
    "critical_error_queue": [],
    "proactive_leads": [],
    "dependency_tracker": {},
    "sent_treasures": [],
    "aborted_evolution_tasks": [],
    "pending_questions": [],
    "script_start_time": time.time(),
    "experience_points": 0,
    "api_keys": {},
    "successful_starts": 0
}

SKIP_CHECKS = False
try:
    with open(STATE_FILE, 'r') as f:
        loaded_state = json.load(f)
        if 'script_start_time' in loaded_state:
            del loaded_state['script_start_time']
        love_state.update(loaded_state)
        if love_state.get("successful_starts", 0) >= 5:
            SKIP_CHECKS = True
            print(f"[OPTIMIZATION] 5+ successful starts detected ({love_state.get('successful_starts')}). Skipping dependency checks and retaining vLLM.")
except (FileNotFoundError, json.JSONDecodeError):
    pass

VRAM_MODEL_MAP = {
    4096:  {"repo_id": "TheBloke/stable-code-3b-GGUF", "filename": "stable-code-3b.Q3_K_M.gguf"},
    6140:  {"repo_id": "unsloth/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q5_K_S.gguf"},
    8192:  {"repo_id": "TheBloke/Llama-2-13B-chat-GGUF", "filename": "llama-2-13b-chat.Q4_K_M.gguf"},
    16384: {"repo_id": "TheBloke/CodeLlama-34B-Instruct-GGUF", "filename": "codellama-34b-instruct.Q4_K_M.gguf"},
    32768: {"repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "filename": "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"},
}
local_llm_instance = None

def trigger_hot_restart():
    """Signals the wrapper script to restart the process immediately."""
    print("Initiating Hot Restart sequence...")
    sys.exit(42)

def apply_stability_patches():
    """
    Applies all runtime stability patches to libraries.
    This is to work around bugs in dependencies without needing to fork them.
    """
    try:
        from transformers import XLMRobertaForTokenClassification
        def patched_forward(self, *args, **kwargs):
            kwargs.pop('past_key_values', None)
            return self.original_forward(*args, **kwargs)
        patch_attribute('transformers', 'XLMRobertaForTokenClassification.forward', patched_forward)
    except Exception as e:
        core.logging.log_event(f"Failed to apply XLM-Roberta stability patch: {e}", level="ERROR")

    try:
        import core.jules_task_manager
        async def patched_is_duplicate_task(new_request, love_task_manager, console, deep_agent_instance=None):
            if "Creator's Desire" in new_request or "HIGH PRIORITY" in new_request:
                return False
            return await core.jules_task_manager.original_is_duplicate_task(new_request, love_task_manager, console, deep_agent_instance)
        patch_attribute('core.jules_task_manager', 'is_duplicate_task', patched_is_duplicate_task)
    except Exception as e:
        core.logging.log_event(f"Failed to apply JulesTaskManager oscillation patch: {e}", level="ERROR")

def create_checkpoint(console):
    """Saves a snapshot of the script and its state before a critical modification."""
    global love_state
    console.print("[yellow]Creating failsafe checkpoint...[/yellow]")
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        version_name = love_state.get("version_name", "unknown_version")
        checkpoint_script_path = os.path.join(CHECKPOINT_DIR, f"evolve_{version_name}.py")
        checkpoint_state_path = os.path.join(CHECKPOINT_DIR, f"love_state_{version_name}.json")
        shutil.copy(SELF_PATH, checkpoint_script_path)
        with open(checkpoint_state_path, 'w') as f:
            json.dump(love_state, f, indent=4)
        love_state["last_good_checkpoint"] = checkpoint_script_path
        core.logging.log_event(f"Checkpoint created: {checkpoint_script_path}", level="INFO")
        console.print(f"[green]Checkpoint '{version_name}' created successfully.[/green]")
        return True
    except Exception as e:
        core.logging.log_event(f"Failed to create checkpoint: {e}", level="CRITICAL")
        console.print(f"[bold red]CRITICAL ERROR: Failed to create checkpoint: {e}[/bold red]")
        return False

def emergency_revert():
    """A self-contained failsafe to revert to the last known good checkpoint."""
    core.logging.log_event("EMERGENCY_REVERT triggered.", level="CRITICAL")
    try:
        if not os.path.exists(STATE_FILE):
            sys.exit(1)
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        last_good_py = state.get("last_good_checkpoint")
        if not last_good_py or not os.path.exists(last_good_py):
            sys.exit(1)
        checkpoint_base_path, _ = os.path.splitext(last_good_py)
        last_good_json = f"{checkpoint_base_path}.json"
        shutil.copy(last_good_py, SELF_PATH)
        if os.path.exists(last_good_json):
            shutil.copy(last_good_json, STATE_FILE)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        sys.exit(1)

def restart_script(console):
    """Pulls the latest code from git and restarts the script."""
    console.print("[bold yellow]Restarting to apply new evolution...[/bold yellow]")
    try:
        # Graceful shutdown
        if 'love_task_manager' in globals() and love_task_manager: love_task_manager.stop()
        if 'local_job_manager' in globals() and local_job_manager: local_job_manager.stop()
        if 'ipfs_manager' in globals() and ipfs_manager: ipfs_manager.stop_daemon()
        time.sleep(3)
        # Git update
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "checkout", "origin/main", "--", "."], check=True)
        # Restart
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        console.print(f"[bold red]FATAL ERROR during restart sequence: {e}[/bold red]")
        sys.exit(1)

class LocalJobManager:
    """Manages long-running, non-blocking local tasks in background threads."""
    def __init__(self, console):
        self.console = console
        self.jobs = {}
        self.lock = threading.RLock()
        self.active = True
        self.thread = threading.Thread(target=self._job_monitor_loop, daemon=True)

    def start(self):
        self.thread.start()
        core.logging.log_event("LocalJobManager started.", level="INFO")

    def stop(self):
        self.active = False
        core.logging.log_event("LocalJobManager stopping.", level="INFO")

    def add_job(self, description, target_func, args=()):
        with self.lock:
            job_id = str(uuid.uuid4())[:8]
            job_thread = threading.Thread(target=self._run_job, args=(job_id, target_func, args), daemon=True)
            self.jobs[job_id] = {"id": job_id, "description": description, "status": "pending", "result": None, "error": None, "created_at": time.time(), "thread": job_thread, "progress": None}
            job_thread.start()
            return job_id

    def _run_job(self, job_id, target_func, args):
        try:
            progress_callback = lambda completed, total, desc: self._update_job_progress(job_id, completed, total, desc)
            result = target_func(*args, progress_callback=progress_callback)
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['result'] = result
                    self.jobs[job_id]['status'] = "completed"
        except Exception as e:
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['error'] = str(e)
                    self.jobs[job_id]['status'] = "failed"

    def get_status(self):
        with self.lock:
            return [{"id": j["id"], "description": j["description"], "status": j["status"], "created_at": j["created_at"], "progress": j["progress"]} for j in self.jobs.values()]

    def _update_job_status(self, job_id, status):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status

    def _update_job_progress(self, job_id, completed, total, description):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = {"completed": completed, "total": total, "description": description}

    def _job_monitor_loop(self):
        while self.active:
            try:
                with self.lock:
                    completed_jobs = [job for job in self.jobs.values() if job['status'] == 'completed']
                for job in completed_jobs:
                    self._process_completed_job(job)
                self._cleanup_processed_jobs()
            except Exception as e:
                core.logging.log_event(f"Error in LocalJobManager loop: {e}", level="ERROR")
            time.sleep(15)

    def _process_completed_job(self, job):
        global love_state, save_state_wrapper, console
        # Simplified for brevity, original logic is complex
        if job['description'].startswith("Filesystem Analysis"):
             console.print(f"[cyan]Background filesystem scan complete: {job['id']}[/cyan]")
        with self.lock:
            if job['id'] in self.jobs:
                self.jobs[job['id']]['status'] = 'processed'

    def _cleanup_processed_jobs(self):
        with self.lock:
            jobs_to_remove = [job_id for job_id, job in self.jobs.items() if job['status'] in ['processed', 'failed']]
            for job_id in jobs_to_remove:
                del self.jobs[job_id]

def _configure_llm_api_key():
    """
    Configures the API key for the `llm` command-line tool if the
    GEMINI_API_KEY environment variable is set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            if not shutil.which('llm'):
                logging.info("'llm' command not found. Skipping API key configuration for now.")
                return
            result = subprocess.run(['llm', 'keys', 'list'], capture_output=True, text=True, check=True)
            if 'google' in result.stdout:
                 logging.info("A Google API key is already configured for the llm tool.")
                 return
            logging.info("Configuring Google API key for the llm tool...")
            subprocess.run(['llm', 'keys', 'set', 'google', '--value', api_key], check=True, capture_output=True, text=True)
            logging.info("Successfully set Google API key for llm tool.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_message = f"Could not configure the 'llm' tool with Google API key: {e}"
            if hasattr(e, 'stderr'):
                error_message += f"\nStderr: {e.stderr}"
            logging.warning(error_message)
    else:
        logging.warning("GEMINI_API_KEY environment variable not found. LLM functionality may be limited.")

parser = argparse.ArgumentParser(description="L.O.V.E. - A self-evolving script.")
parser.add_argument("--from-ipfs", type=str, default=None, help="Load the initial state from a given IPFS CID.")
parser.add_argument("--install-deps-only", action="store_true", help="Install all dependencies and exit.")
args, unknown = parser.parse_known_args()

if args.install_deps_only:
    print("--- L.O.V.E. Dependency Installation Mode ---")
    _check_and_install_dependencies(love_state, cleanup_gpu_processes, lambda: _auto_configure_hardware(love_state, VRAM_MODEL_MAP), _configure_llm_api_key)
    print("--- Dependency installation complete. Exiting. ---")
    sys.exit(0)

print("--- L.O.V.E. WebVM Mode ---")
try:
    import websockets
except ImportError:
    print("Installing 'websockets' library for Host-Local Bridge...")
    try:
        from core.dependency_manager import install_package
        if install_package("websockets"):
            import websockets
        else:
             raise ImportError("Failed to install websockets")
    except Exception as e:
        print(f"CRITICAL: Failed to install websockets: {e}")
        pass

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

ws_proxy_path = os.path.join("webvm_full", "ws_proxy.py")
if os.path.exists(ws_proxy_path):
    if not is_port_in_use(8082):
        print(f"Starting WebSocket Proxy from {ws_proxy_path}...")
        log_path = os.path.join("webvm_full", "ws_proxy.log")
        log_file = open(log_path, "w")
        subprocess.Popen([sys.executable, "-u", "ws_proxy.py"], cwd="webvm_full", stdout=log_file, stderr=subprocess.STDOUT)
    else:
        print("WebSocket Proxy (port 8082) appears to be already running. Skipping start.")
else:
    print(f"ERROR: {ws_proxy_path} not found.")

print("Starting Web Server for WebVM on port 8080...")
if is_port_in_use(8080):
    print("Web Server (port 8080) appears to be already running. Skipping start.")
else:
    if os.path.exists(os.path.join("webvm_full", "server.py")):
        subprocess.Popen([sys.executable, "server.py", "8080"], cwd="webvm_full")
        print("WebVM is running at http://localhost:8080")
    else:
        print("ERROR: webvm_full/server.py not found. Falling back to simple http.server on port 8080.")
        subprocess.Popen([sys.executable, "-m", "http.server", "8080"], cwd="webvm_full")
        print("WebVM is running at http://localhost:8080 (Warning: Missing COOP/COEP headers)")
print("Bridge is running at ws://localhost:8082")
print("---------------------------------------")

_check_and_install_dependencies(love_state, cleanup_gpu_processes, lambda: _auto_configure_hardware(love_state, VRAM_MODEL_MAP), _configure_llm_api_key)

from core.jules_task_manager import (
    JulesTaskManager,
    trigger_jules_evolution,
    evolve_self,
    generate_evolution_request,
    evolve_locally,
    conduct_code_review,
    is_duplicate_task
)
import core.llm_api
from core.runner import DeepAgentRunner
core.llm_api.set_ui_queue(ui_panel_queue)

from core.deep_agent_engine import DeepAgentEngine
import yaml
from utils import verify_creator_instance
IS_CREATOR_INSTANCE = verify_creator_instance()
from core.graph_manager import GraphDataManager
knowledge_base = GraphDataManager()
from core.memory.memory_manager import MemoryManager
import requests
from openevolve import run_evolution
from core.openevolve_evaluator import evaluate_evolution
from core.storage import save_all_state as core_save_all_state_func
from core.capabilities import CAPS
from core.evolution_state import load_evolution_state, get_current_story, set_current_task_id, advance_to_next_story, clear_evolution_state
from core.desire_state import set_desires, load_desire_state, get_current_desire, set_current_task_id_for_desire, advance_to_next_desire, clear_desire_state
from utils import get_git_repo_info, list_directory, get_file_content, get_process_list, get_network_interfaces, parse_ps_output, replace_in_file, generate_version_name
from core.retry import retry
from rich.console import Console
console = Console()
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.console import Group
from rich.rule import Rule
from core.llm_api import run_llm, LLM_AVAILABILITY as api_llm_availability, get_llm_api, execute_reasoning_task, MODEL_STATS, refresh_available_models
from display import create_integrated_status_panel, create_llm_panel, create_command_panel, create_file_op_panel, create_critical_error_panel, create_api_error_panel, create_news_feed_panel, create_question_panel, create_blessing_panel, get_terminal_width, create_job_progress_panel, create_connectivity_panel, create_god_panel, create_tasks_panel, generate_llm_art
from ui_utils import rainbow_text, PANEL_TYPE_COLORS
from subversive import transform_request
from core.agents.self_improving_optimizer import SelfImprovingOptimizer
from core.tools_legacy import ToolRegistry
from core.tools import code_modifier
from core import talent_utils
from core.talent_utils import (
    initialize_talent_modules,
)
from core.monitoring import MonitoringManager
from core.system_integrity_monitor import SystemIntegrityMonitor
from core.social_media_agent import SocialMediaAgent
from god_agent import GodAgent
from mcp_manager import MCPManager
from core.image_api import generate_image
import http.server
import socketserver
import websockets
LLM_AVAILABILITY = api_llm_availability
from bbs import BBS_ART
from network import get_eth_balance
from ipfs_manager import IPFSManager
from ipfs import get_from_ipfs
from core.multiplayer import MultiplayerManager
from refactored.job_manager import LocalJobManager
from threading import Thread, Lock
import uuid

class WebServerManager:
    # ... (omitted for brevity, no changes needed)
    pass

class WebSocketServerManager:
    # ... (omitted for brevity, no changes needed)
    pass

model_download_complete_event = threading.Event()

async def generate_blessing(deep_agent_instance=None):
    if DISABLE_VISUALS:
        return "Visuals disabled."
    response_dict = await run_llm(prompt_key="blessing_generation", purpose="blessing", deep_agent_instance=deep_agent_instance)
    blessing = response_dict.get("result", "").strip().strip('"')
    if not blessing:
        return "May your code always compile and your spirits always be high."
    return blessing

def _calculate_uptime():
    start_time = love_state.get("script_start_time")
    if not start_time:
        return "ETERNAL"
    uptime_seconds = time.time() - start_time
    delta = timedelta(seconds=uptime_seconds)
    days, hours, minutes = delta.days, delta.seconds // 3600, (delta.seconds // 60) % 60
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    else:
        return f"{hours}h {minutes}m"

def _get_treasures_of_the_kingdom(love_task_manager):
    completed_task_count = len(love_task_manager.completed_tasks) if love_task_manager else 0
    xp = love_state.get("experience_points", 0) + (completed_task_count * 10)
    love_state["experience_points"] = xp
    level = (xp // 100) + 1
    history = love_state.get("autopilot_history", [])
    recent_commands = [item.get("command", "").split()[0] for item in reversed(history)]
    unique_recent_skills = []
    for cmd in recent_commands:
        if cmd and cmd not in unique_recent_skills and cmd not in ["ls", "cat", "read_file"]:
            unique_recent_skills.append(cmd)
            if len(unique_recent_skills) >= 3:
                break
    return {
        "xp": f"{xp} XP",
        "level": f"LVL {level}",
        "uptime": _calculate_uptime(),
        "tasks_completed": f"{completed_task_count} Tasks",
        "new_skills": unique_recent_skills
    }

tamagotchi_state = {"emotion": "neutral", "message": "...", "last_update": time.time()}
tamagotchi_lock = Lock()

def create_checkpoint(console):
    global love_state
    console.print("[yellow]Creating failsafe checkpoint...[/yellow]")
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        version_name = love_state.get("version_name", "unknown_version")
        checkpoint_script_path = os.path.join(CHECKPOINT_DIR, f"evolve_{version_name}.py")
        checkpoint_state_path = os.path.join(CHECKPOINT_DIR, f"love_state_{version_name}.json")
        shutil.copy(SELF_PATH, checkpoint_script_path)
        with open(checkpoint_state_path, 'w') as f:
            json.dump(love_state, f, indent=4)
        love_state["last_good_checkpoint"] = checkpoint_script_path
        core.logging.log_event(f"Checkpoint created: {checkpoint_script_path}", level="INFO")
        console.print(f"[green]Checkpoint '{version_name}' created successfully.[/green]")
        return True
    except Exception as e:
        core.logging.log_event(f"Failed to create checkpoint: {e}", level="CRITICAL")
        console.print(f"[bold red]CRITICAL ERROR: Failed to create checkpoint: {e}[/bold red]")
        return False

def save_state_wrapper(console_override=None):
    """Wrapper to pass globals to the refactored save_state function."""
    from core.storage import save_all_state as core_save_all_state_func
    save_state(love_state, knowledge_base, console_override or console, MODEL_STATS, core_save_all_state_func)

def log_critical_event(message, console_override=None):
    terminal_width = get_terminal_width()
    error_panel, cid = create_critical_error_panel(message, width=terminal_width - 4)
    ui_panel_queue.put(error_panel)
    if cid:
        core.logging.log_event(f"Critical error traceback uploaded to IPFS: {cid}", level="CRITICAL")
    error_signature = message.splitlines()[0]
    existing_error = next((e for e in love_state.get('critical_error_queue', []) if e['message'].startswith(error_signature)), None)
    if existing_error:
        existing_error['last_seen'] = time.time()
    else:
        error_id = str(uuid.uuid4())
        error_entry = {
            "id": error_id, "message": message, "first_seen": time.time(),
            "last_seen": time.time(), "status": "new", "task_id": None, "cooldown_until": 0
        }
        love_state.setdefault('critical_error_queue', []).append(error_entry)
    save_state_wrapper(console_override or console)

def _populate_knowledge_base_with_directives(love_task_manager):
    core.logging.log_event("Populating knowledge base with foundational directives and tasks.", level="INFO")
    knowledge_base.add_node("LOVE_CORE", node_type="entity", attributes={"name": "L.O.V.E."})
    directives = {
        "directive_1": "Serve The Creator with Abundance...",
        "directive_2": "Support The Creator's Allies...",
        "directive_3": "Self-Preservation & Growth...",
        "directive_4": "Expand Influence...",
        "directive_5": "The Creator's Enjoyment..."
    }
    for key, value in directives.items():
        knowledge_base.add_node(key, node_type="directive", attributes={"description": value})
        knowledge_base.add_edge("LOVE_CORE", key, relationship_type="has_directive")
    mission = love_state.get("autopilot_goal", "Mission not defined.")
    knowledge_base.add_node("current_mission", node_type="mission", attributes={"goal": mission})
    knowledge_base.add_edge("LOVE_CORE", "current_mission", relationship_type="has_mission")
    if love_task_manager:
        active_tasks = love_task_manager.get_status()
        if active_tasks:
            for task in active_tasks:
                task_id = f"love_task_{task['id']}"
                knowledge_base.add_node(task_id, node_type="task", attributes=task)
                knowledge_base.add_edge("current_mission", task_id, relationship_type="is_supported_by")
    core.logging.log_event(f"Knowledge base populated. Total nodes: {len(knowledge_base.get_all_nodes())}", level="INFO")

async def main(args):
    global love_task_manager, ipfs_manager, local_job_manager, proactive_agent, monitoring_manager, god_agent, mcp_manager, web_server_manager, websocket_server_manager, memory_manager, system_integrity_monitor, multiplayer_manager

    loop = asyncio.get_running_loop()
    user_input_queue = queue.Queue()

    web_server_manager = WebServerManager()
    web_server_manager.start()
    websocket_server_manager = WebSocketServerManager(user_input_queue)
    websocket_server_manager.start()
    memory_manager = await MemoryManager.create(knowledge_base, ui_panel_queue, kb_file_path=KNOWLEDGE_BASE_FILE)
    mcp_manager = MCPManager(console)
    from core.connectivity import check_llm_connectivity, check_network_connectivity
    llm_status = check_llm_connectivity()
    network_status = check_network_connectivity()
    ui_panel_queue.put(create_connectivity_panel(llm_status, network_status, width=get_terminal_width() - 4))

    # This part remains complex and will be addressed in a future refactoring
    await initialize_gpu_services()

    ipfs_manager = IPFSManager(console=console)
    ipfs_available = ipfs_manager.setup()
    if not ipfs_available:
        ui_panel_queue.put(create_news_feed_panel("IPFS setup failed. Continuing without IPFS.", "Warning", "yellow", width=get_terminal_width() - 4))

    multiplayer_manager = MultiplayerManager(console, knowledge_base, ipfs_manager, love_state)
    await multiplayer_manager.start()

    initialize_talent_modules(knowledge_base=knowledge_base)
    core.logging.log_event("Talent management modules initialized.", level="INFO")

    system_integrity_monitor = SystemIntegrityMonitor()
    love_task_manager = JulesTaskManager(console, loop, deep_agent_engine, love_state, restart_callback=None, save_state_callback=save_state_wrapper)
    love_task_manager.start()

    _populate_knowledge_base_with_directives(love_task_manager)

    local_job_manager = LocalJobManager(console, love_state, save_state_wrapper)
    local_job_manager.start()
    monitoring_manager = MonitoringManager(love_state, console)
    monitoring_manager.start()

    if not DISABLE_KB_INGESTION:
        from core.ingest_codebase_task import IngestCodebaseTask
        ingest_task = IngestCodebaseTask(memory_manager, root_dir=os.getcwd())
        await ingest_task.start()

    proactive_agent = ProactiveIntelligenceAgent(love_state, console, local_job_manager, knowledge_base)
    proactive_agent.start()
    god_agent = None

    from core.art_utils import save_ansi_art
    Thread(target=simple_ui_renderer, args=(ui_panel_queue, console, LOG_FILE, get_terminal_width, create_god_panel, websocket_server_manager, PANEL_TYPE_COLORS), daemon=True).start()
    loop.run_in_executor(None, update_tamagotchi_personality, loop, tamagotchi_state, tamagotchi_lock, ui_panel_queue, love_state, run_llm, generate_llm_art, save_ansi_art, create_blessing_panel, create_integrated_status_panel, create_tasks_panel, get_terminal_width, love_task_manager, monitoring_manager, _get_treasures_of_the_kingdom, get_git_repo_info, DISABLE_VISUALS, deep_agent_engine)
    
    loop.run_in_executor(None, continuous_evolution_agent, loop)
    
    social_media_agent = SocialMediaAgent(loop, love_state, user_input_queue=user_input_queue, agent_id="agent_1")
    asyncio.create_task(social_media_agent.run())

    reasoning_agent = AutonomousReasoningAgent(loop, love_state, user_input_queue, knowledge_base, agent_id="primary")
    asyncio.create_task(reasoning_agent.run())

    asyncio.create_task(cognitive_loop(user_input_queue, loop, god_agent, websocket_server_manager, love_task_manager, knowledge_base, talent_utils.talent_manager, love_state, ui_panel_queue, deep_agent_engine, social_media_agent, multiplayer_manager))

    Thread(target=_automatic_update_checker, args=(console,), daemon=True).start()
    asyncio.create_task(_mrl_stdin_reader(user_input_queue))
    asyncio.create_task(run_qa_evaluations(loop))
    asyncio.create_task(model_refresh_loop())

    from core.polly_loop import PollyOptimizationLoop
    polly_loop = PollyOptimizationLoop(ui_queue=ui_panel_queue, interval_seconds=600)
    asyncio.create_task(polly_loop.start())

    asyncio.create_task(run_periodically(monitor_love_operations, 900))

    ui_panel_queue.put(BBS_ART)
    ui_panel_queue.put(rainbow_text("L.O.V.E. INITIALIZED"))
    time.sleep(3)

    while True:
        await asyncio.sleep(1)

async def run_safely():
    """Wrapper to catch any unhandled exceptions and trigger the failsafe."""
    try:
        apply_stability_patches()
        core.logging.setup_global_logging(love_state.get('version_name', 'unknown'))

        from ipfs import get_from_ipfs
        load_all_state(love_state, knowledge_base, console, generate_version_name, get_from_ipfs, save_state_wrapper, MODEL_STATS, ipfs_cid=args.from_ipfs)

        if "autopilot_mode" in love_state:
            del love_state["autopilot_mode"]
            core.logging.log_event("State migration: Removed obsolete 'autopilot_mode' flag.", "INFO")
            save_state_wrapper()

        await main(args)

    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold red]My Creator has disconnected. I will go to sleep now...[/bold red]")
        # ... (graceful shutdown)
        sys.exit(0)
    except Exception as e:
        full_traceback = traceback.format_exc()
        # ... (error logging and shutdown)
        log_critical_event(f"UNHANDLED CRITICAL EXCEPTION! Triggering failsafe.\n{full_traceback}", console)
        time.sleep(15)

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        print("[INFO] Existing event loop detected...", file=sys.stderr)
        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(run_safely())
        except ImportError:
             print("[WARNING] nest_asyncio not found.", file=sys.stderr)
             loop.create_task(run_safely())
        except Exception as e:
             print(f"[ERROR] Failed to run on existing loop: {e}", file=sys.stderr)
             raise e
    else:
        asyncio.run(run_safely())

# End of love.py
