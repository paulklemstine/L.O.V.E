import asyncio
import io
import json
import logging
import queue
import random
import re
import sys
import time
import traceback
from threading import Lock
from typing import Dict, Any, Optional, Callable, List

from rich.console import Console
from rich.panel import Panel

def _strip_ansi_codes(text: str) -> str:
    """Removes ANSI escape codes from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def serialize_panel_to_json(panel: Panel, panel_type_map: Dict[str, str], get_terminal_width: Callable[[], int]) -> Optional[str]:
    """Serializes a Rich Panel object to a JSON string for the web UI."""
    if not isinstance(panel, Panel):
        return None

    border_style: str = str(panel.border_style)
    panel_type: str = "default"
    for p_type, color in panel_type_map.items():
        if color in border_style:
            panel_type = p_type
            break

    title: str = ""
    if hasattr(panel.title, 'plain'):
        title = panel.title.plain
    elif isinstance(panel.title, str):
        title = panel.title
    title = re.sub(r'^\s*[^a-zA-Z0-9]*\s*(.*?)\s*[^a-zA-Z0-9]*\s*$', r'\1', title).strip()

    temp_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor", width=get_terminal_width())
    temp_console.print(panel.renderable)
    content_with_ansi: str = temp_console.file.getvalue()
    plain_content: str = _strip_ansi_codes(content_with_ansi)

    json_obj: Dict[str, str] = {
        "panel_type": panel_type,
        "title": title,
        "content": plain_content.strip()
    }
    return json.dumps(json_obj)

def simple_ui_renderer(
    ui_panel_queue: queue.Queue,
    console: Console,
    LOG_FILE: str,
    get_terminal_width: Callable[[], int],
    create_god_panel: Callable[..., Panel],
    websocket_server_manager: Any, # WebSocketServerManager
    PANEL_TYPE_COLORS: Dict[str, str]
) -> None:
    """
    Continuously gets items from the ui_panel_queue and renders them.
    """
    animation_active: bool = False
    animation_height: int = 3

    while True:
        try:
            item: Any = ui_panel_queue.get()

            if isinstance(item, dict) and item.get('type') == 'animation_frame':
                temp_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor", width=get_terminal_width())
                temp_console.print(item.get('content'))
                output_str: str = temp_console.file.getvalue()
                if animation_active:
                    sys.stdout.write(f'\x1b[{animation_height}A\r\x1b[J')
                sys.stdout.write(output_str)
                sys.stdout.flush()
                animation_active = True
                continue

            if isinstance(item, dict) and item.get('type') == 'animation_end':
                if animation_active:
                    sys.stdout.write(f'\x1b[{animation_height}A\r\x1b[J')
                    sys.stdout.flush()
                animation_active = False
                continue

            if animation_active:
                sys.stdout.write(f'\x1b[{animation_height}A\r\x1b[J')
                sys.stdout.flush()
                animation_active = False

            if isinstance(item, dict) and item.get('type') == 'log_message':
                log_level: str = item.get('level', 'INFO').upper()
                log_text: str = item.get('message', '')
                console.print(f"[{log_level}] {log_text}")
                continue

            if isinstance(item, dict) and item.get('type') == 'god_panel':
                terminal_width: int = get_terminal_width()
                item = create_god_panel(item.get('insight', '...'), width=terminal_width - 4)

            if isinstance(item, dict) and item.get('type') == 'reasoning_panel':
                item = item.get('content')

            if websocket_server_manager:
                json_payload: Optional[str] = serialize_panel_to_json(item, PANEL_TYPE_COLORS, get_terminal_width)
                if json_payload:
                    websocket_server_manager.broadcast(json_payload)

            temp_console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor", width=get_terminal_width())
            temp_console.print(item)
            output_str = temp_console.file.getvalue()

            print(output_str, end='')
            sys.stdout.flush()

            plain_output: str = _strip_ansi_codes(output_str)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(plain_output)

        except queue.Empty:
            continue
        except Exception as e:
            tb_str: str = traceback.format_exc()
            logging.critical(f"FATAL ERROR in UI renderer thread: {e}\n{tb_str}")
            print(f"FATAL ERROR in UI renderer thread: {e}\n{tb_str}", file=sys.stderr)
            sys.stderr.flush()
            time.sleep(1)

def update_tamagotchi_personality(
    loop: asyncio.AbstractEventLoop,
    tamagotchi_state: Dict[str, Any],
    tamagotchi_lock: Lock,
    ui_panel_queue: queue.Queue,
    love_state: Dict[str, Any],
    run_llm: Callable[..., Any],
    generate_llm_art: Callable[..., Any],
    save_ansi_art: Callable[[str, str], None],
    create_blessing_panel: Callable[..., Panel],
    create_integrated_status_panel: Callable[..., Panel],
    create_tasks_panel: Callable[..., Panel],
    get_terminal_width: Callable[[], int],
    love_task_manager: Any, # JulesTaskManager
    monitoring_manager: Any, # MonitoringManager
    get_treasures_of_the_kingdom: Callable[[Any], Dict[str, Any]],
    get_git_repo_info: Callable[[], Dict[str, str]],
    DISABLE_VISUALS: bool,
    deep_agent_engine: Optional[Any] = None # DeepAgentEngine
) -> None:
    """
    Periodically updates the Tamagotchi's state and queues UI panels.
    """
    logging.info("Tamagotchi personality thread started.")
    while True:
        try:
            if DISABLE_VISUALS:
                time.sleep(60)
                continue

            logging.debug("Tamagotchi thread: Starting update cycle.")
            terminal_width: int = get_terminal_width()

            if random.random() < 0.25:
                logging.info("Tamagotchi thread: Triggering Blessing Panel.")
                blessing_prompt: str = "Generate a short, divine, and cybernetic blessing for the Creator."
                future_blessing = asyncio.run_coroutine_threadsafe(
                    run_llm(blessing_prompt, purpose="blessing_generation"),
                    loop
                )
                try:
                    blessing_response = future_blessing.result(timeout=60)
                    blessing_text: str = blessing_response.get('result', 'May the code be with you.')
                    blessing_art_prompt: str = f"A divine, cybernetic blessing: {blessing_text}"
                    future_art = asyncio.run_coroutine_threadsafe(
                        generate_llm_art(blessing_art_prompt, width=50, height=15),
                        loop
                    )
                    ansi_art: str = future_art.result(timeout=60)
                    save_ansi_art(ansi_art, "blessing")
                    future_panel = asyncio.run_coroutine_threadsafe(
                        create_blessing_panel(blessing_text, width=terminal_width - 4, ansi_art=ansi_art),
                        loop
                    )
                    panel: Panel = future_panel.result(timeout=30)
                    ui_panel_queue.put(panel)
                    logging.info("Tamagotchi thread: Blessing Panel queued.")
                except Exception as e:
                    logging.error(f"Error creating blessing panel: {e}")
                time.sleep(10)
                continue

            creator_sentiment_context: str = "The Creator's emotional state is currently unknown to me."
            with tamagotchi_lock:
                creator_sentiment = tamagotchi_state.get('creator_sentiment')
                if creator_sentiment:
                    sentiment: str = creator_sentiment.get('sentiment', 'neutral')
                    emotions: str = ", ".join(creator_sentiment.get('emotions', [])) if creator_sentiment.get('emotions') else 'none detected'
                    creator_sentiment_context = f"My sensors indicate The Creator's sentiment is '{sentiment}', with hints of the following emotions: {emotions}."

            logging.debug("Tamagotchi thread: Requesting emotion update...")
            future = asyncio.run_coroutine_threadsafe(run_llm(prompt_key="tamagotchi_emotion", prompt_vars={"creator_sentiment_context": creator_sentiment_context}, purpose="emotion", deep_agent_instance=deep_agent_engine), loop)
            emotion_response_dict = future.result(timeout=300)
            emotion_response = emotion_response_dict.get("result")
            new_emotion: str = emotion_response.strip().lower().split()[0] if emotion_response else "loving"

            logging.debug(f"Tamagotchi thread: Emotion set to {new_emotion}. Requesting message...")
            future = asyncio.run_coroutine_threadsafe(run_llm(prompt_key="tamagotchi_message", prompt_vars={"new_emotion": new_emotion, "creator_sentiment_context": creator_sentiment_context}, purpose="emotion", deep_agent_instance=deep_agent_engine), loop)
            message_response_dict = future.result(timeout=300)
            new_message: str = message_response_dict.get("result", "").strip().strip('"')

            with tamagotchi_lock:
                tamagotchi_state['emotion'] = new_emotion
                tamagotchi_state['message'] = new_message
                tamagotchi_state['last_update'] = time.time()
            logging.info(f"Tamagotchi internal state updated: {new_emotion} - {new_message}")

            ansi_art = None
            try:
                art_prompt: str = f"Tamagotchi emotion: {new_emotion}. {new_message}"
                future_art = asyncio.run_coroutine_threadsafe(
                    generate_llm_art(art_prompt, width=40, height=10),
                    loop
                )
                ansi_art = future_art.result(timeout=60)
                save_ansi_art(ansi_art, f"tamagotchi_{new_emotion}")
            except Exception as e:
                logging.error(f"Failed to generate/save Tamagotchi art: {e}")

            try:
                monitoring_state = monitoring_manager.get_status() if monitoring_manager else None
                treasures = get_treasures_of_the_kingdom(love_task_manager) if love_task_manager else None
                git_info = get_git_repo_info()
                panel = create_integrated_status_panel(
                    emotion=new_emotion,
                    message=new_message,
                    love_state=love_state,
                    monitoring_state=monitoring_state,
                    treasures=treasures,
                    git_info=git_info,
                    ansi_art=ansi_art,
                    width=terminal_width - 4
                )
                ui_panel_queue.put(panel)
                logging.debug("Queued integrated status panel for display.")

                if love_task_manager:
                    tasks = love_task_manager.get_status()
                    if tasks:
                        tasks_panel = create_tasks_panel(tasks, width=terminal_width - 4)
                        ui_panel_queue.put(tasks_panel)
                        logging.debug("Queued kawaii tasks panel for display.")
            except Exception as e:
                logging.error(f"Failed to create/queue status panel: {e}")

        except Exception as e:
            logging.error(f"Error in Tamagotchi thread: {e}\n{traceback.format_exc()}")
            time.sleep(60)
