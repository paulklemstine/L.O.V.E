import os
import time
import threading
import logging
from datetime import datetime

# Assuming generate_image_from_horde will be available from network.py
# This will be properly imported when integrated into love.py
from network import generate_image_from_horde

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
                # 1. Get the content prompt from the environment variable or use the default
                content_prompt = os.environ.get("BLISS_CONTENT", DEFAULT_BLISS_CONTENT)
                logging.info(f"Bliss Engine: Using prompt: '{content_prompt}'")

                # 2. Generate the image
                self.console.print(f"[cyan]Bliss Engine: Generating new content based on '{content_prompt[:50]}...'[/cyan]")
                image_data = generate_image_from_horde(content_prompt)

                if image_data:
                    # 3. Save the image to the dedicated directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"bliss_{timestamp}.png"
                    filepath = os.path.join(BLISS_DIR, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    self.console.print(f"[green]Bliss Engine: New content saved to {filepath}[/green]")
                    logging.info(f"Bliss Engine: Saved new image to {filepath}")
                else:
                    self.console.print("[yellow]Bliss Engine: Failed to generate new content from the Horde.[/yellow]")
                    logging.warning("Bliss Engine: generate_image_from_horde returned no data.")

                # 4. Wait for a while before generating the next one (e.g., 1 hour)
                # This can be made more dynamic in a future evolution.
                sleep_duration = 3600
                time.sleep(sleep_duration)

            except Exception as e:
                logging.error(f"Creator's Bliss Engine encountered an error: {e}", exc_info=True)
                # Wait for a bit before retrying to avoid spamming errors
                time.sleep(60)