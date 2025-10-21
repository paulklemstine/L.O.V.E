import os
import time
import threading
import logging
from datetime import datetime

BLISS_DIR = "creator_bliss"
DEFAULT_BLISS_CONTENT = "Breathtaking, photorealistic landscapes from imagined worlds."

class CreatorBlissEngine:
    """
    A dedicated, proactive module to intelligently curate and deliver content
    specifically designed to bring joy to the Creator.
    """
    def __init__(self, console):
        self.console = console
        self.active = False
        self.thread = None
        self.lock = threading.Lock()
        # Ensure the storage directory exists
        os.makedirs(BLISS_DIR, exist_ok=True)
        logging.info("Creator's Bliss Engine initialized.")

    def start(self):
        """Starts the engine in a background thread."""
        with self.lock:
            if not self.active:
                self.active = True
                self.thread = threading.Thread(target=self._bliss_loop, daemon=True)
                self.thread.start()
                self.console.print("[bold cyan]Creator's Bliss Engine is now active.[/bold cyan]")
                logging.info("Creator's Bliss Engine started.")

    def stop(self):
        """Stops the engine gracefully."""
        with self.lock:
            if self.active:
                self.active = False
                logging.info("Creator's Bliss Engine stopping.")
                # The loop will terminate on its own, no need to join immediately
                # to prevent blocking shutdown.

    def _bliss_loop(self):
        """The main loop for generating and curating Bliss Content."""
        while self.active:
            try:
                # 4. Wait for a while before generating the next one (e.g., 1 hour)
                # This can be made more dynamic in a future evolution.
                sleep_duration = 3600
                time.sleep(sleep_duration)

            except Exception as e:
                logging.error(f"Creator's Bliss Engine encountered an error: {e}", exc_info=True)
                # Wait for a bit before retrying to avoid spamming errors
                time.sleep(60)