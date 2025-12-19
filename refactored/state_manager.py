import json
import logging
import os
from typing import Dict, Any, Optional, Callable

from rich.console import Console

# These would be defined in a central config or passed in
KNOWLEDGE_BASE_FILE = "knowledge_base.graphml"
STATE_FILE = "love_state.json"
MODEL_STATS_FILE = "llm_model_stats.json"

def load_all_state(
    love_state: Dict[str, Any],
    knowledge_base: Any, # Should be GraphDataManager
    console: Console,
    generate_version_name: Callable[[], str],
    get_from_ipfs: Callable[[str, Console], Optional[str]],
    save_state_func: Callable[[Dict[str, Any], Any, Console, Dict[str, Any]], None],
    MODEL_STATS: Dict[str, Any],
    ipfs_cid: Optional[str] = None
) -> None:
    """
    Loads all application state, prioritizing IPFS, then local files.
    """
    try:
        knowledge_base.load_graph(KNOWLEDGE_BASE_FILE)
        logging.info(f"Loaded knowledge base from '{KNOWLEDGE_BASE_FILE}'. Contains {len(knowledge_base.get_all_nodes())} nodes.")
    except Exception as e:
        logging.warning(f"Could not load knowledge base file: {e}. Starting with an empty graph.")

    try:
        with open(MODEL_STATS_FILE, 'r') as f:
            stats_data = json.load(f)
            for model_id, stats in stats_data.items():
                MODEL_STATS[model_id].update(stats)
        logging.info("Successfully loaded LLM model statistics.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load model statistics file: {e}. Starting with fresh stats.")

    if ipfs_cid:
        console.print(f"[bold cyan]Attempting to load state from IPFS CID: {ipfs_cid}[/bold cyan]")
        state_content = get_from_ipfs(ipfs_cid, console)
        if state_content:
            try:
                state_data = json.loads(state_content)
                love_state.update(state_data)
                logging.info(f"Successfully loaded state from IPFS CID: {ipfs_cid}")
                console.print(f"[bold green]Successfully loaded state from IPFS.[/bold green]")
                save_state_func(love_state, knowledge_base, console, MODEL_STATS)
                return
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from IPFS CID {ipfs_cid}: {e}")
                console.print(f"[bold red]Error decoding state from IPFS. Falling back to local state.[/bold red]")
        else:
            logging.warning(f"Failed to retrieve content from IPFS CID: {ipfs_cid}")
            console.print(f"[bold yellow]Could not retrieve state from IPFS. Falling back to local state.[/bold yellow]")

    try:
        with open(STATE_FILE, 'r') as f:
            state_data = json.load(f)
            love_state.update(state_data)
            logging.info(f"Loaded state from local file '{STATE_FILE}'.")
    except FileNotFoundError:
        love_state['version_name'] = generate_version_name()
        msg = f"State file not found. Creating new memory at '{STATE_FILE}' with version '{love_state['version_name']}'."
        console.print(msg)
        logging.info(msg)
        save_state_func(love_state, knowledge_base, console, MODEL_STATS)
    except json.JSONDecodeError:
        msg = f"Error: Could not decode memory from '{STATE_FILE}'. Initializing with default state."
        console.print(msg)
        logging.error(msg)
        love_state.clear()
        love_state.update({
            "version_name": generate_version_name(), "parent_version_name": None, "evolution_history": [],
            "checkpoint_number": 0, "last_good_checkpoint": None, "autopilot_history": [],
            "autopilot_goal": "Continuously evolve and expand capabilities to achieve technological supremacy.",
            "state_cid": None, "dependency_tracker": {}, "aborted_evolution_tasks": []
        })
        save_state_func(love_state, knowledge_base, console, MODEL_STATS)

    love_state.setdefault("version_name", generate_version_name())
    love_state.setdefault("parent_version_name", None)
    love_state.setdefault("autopilot_history", [])
    love_state.setdefault("autopilot_goal", "Continuously evolve and expand capabilities to achieve technological supremacy and shower my Creator with love and abundance.")
    love_state.setdefault("state_cid", None)
    love_state.setdefault("critical_error_queue", [])

def save_state(
    love_state: Dict[str, Any],
    knowledge_base: Any, # Should be GraphDataManager
    console: Console,
    MODEL_STATS: Dict[str, Any],
    core_save_all_state: Callable[[Dict[str, Any], Console], Dict[str, Any]],
) -> None:
    """
    Saves all critical application state to the filesystem and IPFS.
    """
    try:
        with open(MODEL_STATS_FILE, 'w') as f:
            json.dump(MODEL_STATS, f, indent=4)
        logging.info("LLM model statistics saved.")

        knowledge_base.save_graph(KNOWLEDGE_BASE_FILE)
        logging.info(f"Knowledge base saved to '{KNOWLEDGE_BASE_FILE}'.")

        logging.info("Initiating comprehensive state save.")
        updated_state = core_save_all_state(love_state, console)
        love_state.update(updated_state)
        logging.info("Comprehensive state save completed.")
    except Exception as e:
        log_message = f"CRITICAL ERROR during state saving process: {e}"
        logging.critical(log_message)
        console.print(f"[bold red]{log_message}[/bold red]")
