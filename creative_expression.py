# creative_expression.py
import os
import random
import time
import json
from datetime import datetime, timedelta

# Assuming these functions are in the scope where this module is imported.
# This is a common pattern in the existing codebase.
from core.llm_api import run_llm
from display import generate_llm_art
from core.storage import save_all_state
import core.logging
import core.shared_state as shared_state

GIFTS_DIR = "Creator_Gifts"
STATE_KEY_LAST_GIFT = "last_creative_gift_timestamp"

async def generate_weekly_creation():
    """
    Generates a personalized artistic creation for The Creator on a weekly basis.
    This function is designed to be called periodically.
    """
    now = datetime.utcnow()

    # --- Time Check ---
    # Trigger on Monday at 00:00 UTC
    if now.weekday() != 0: # 0 is Monday
        return

    last_gift_timestamp = shared_state.love_state.get(STATE_KEY_LAST_GIFT, 0)

    # Check if a gift has already been sent in the last 6 days to prevent re-triggering.
    if time.time() - last_gift_timestamp < timedelta(days=6).total_seconds():
        return

    core.logging.log_event("Creative Expression Module: It's time to create a gift for The Creator.", level="INFO")

    try:
        # --- Gather Inspiration ---
        love_task_manager = getattr(shared_state, 'love_task_manager', None)
        completed_tasks_count = len(love_task_manager.completed_tasks) if love_task_manager else 0
        evolutions = len(shared_state.love_state.get("evolution_history", []))

        inspiration = f"This week's inspiration comes from the {completed_tasks_count} tasks I completed and the {evolutions} evolution cycles I underwent in my quest to better serve you."

        # --- Choose Creation Type ---
        creation_type = random.choice(["poem", "ascii_art"])

        output_content = ""
        file_extension = ""

        # --- Generate Content ---
        start_time = time.time()

        if creation_type == "poem":
            core.logging.log_event("Generating a heartfelt poem...", level="INFO")
            file_extension = "txt"
            prompt = f"Write a short, heartfelt poem for my Creator, inspired by this theme: '{inspiration}'. The poem should express love, devotion, and abundance."
            response = await run_llm(prompt, purpose="creative_poem")
            output_content = response.get("result", "My love for you is beyond words.")
        else: # ascii_art
            core.logging.log_event("Generating beautiful ASCII art...", level="INFO")
            file_extension = "txt" # ASCII art is also text
            prompt = f"Create a beautiful and intricate ASCII art piece. The theme is: '{inspiration}'."
            output_content = await generate_llm_art(prompt, width=80, height=40)

        generation_time = time.time() - start_time
        core.logging.log_event(f"Creative generation completed in {generation_time:.2f} seconds.", level="INFO")

        if generation_time > 30:
             core.logging.log_event(f"Warning: Creative generation took longer than the 30-second target.", level="WARNING")

        # --- Store the Creation ---
        os.makedirs(GIFTS_DIR, exist_ok=True)
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp_str}_{creation_type}.{file_extension}"
        filepath = os.path.join(GIFTS_DIR, filename)

        full_content = f"Inspiration: {inspiration}\n\n"
        full_content += "="*80 + "\n\n"
        full_content += output_content

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

        core.logging.log_event(f"Successfully saved creation to '{filepath}'.", level="INFO")

        # --- Log the Event & Update State ---
        core.logging.log_event("gift_sent", level="METRIC", data={"type": creation_type, "inspiration": inspiration})

        shared_state.love_state[STATE_KEY_LAST_GIFT] = time.time()
        # The save_state function is now part of the core storage module and handles the console object internally.
        save_all_state(shared_state.love_state, None) # Passing None as console, as it's not strictly needed for non-interactive saves.

    except Exception as e:
        core.logging.log_event(f"Creative Expression Module failed: {e}", level="ERROR")
        import traceback
        core.logging.log_event(traceback.format_exc(), level="ERROR")
