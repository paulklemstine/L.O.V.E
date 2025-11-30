import logging
import sys
import re
import threading
import time
import os

# --- CONFIGURATION & GLOBALS ---
LOG_FILE = "love.log"
log_file_stream = None
ui_panel_queue = None


def initialize_logging_with_ui_queue(queue):
    """
    Initializes the logging system with the UI panel queue.
    This must be called from the main script after the queue is created.
    """
    global ui_panel_queue
    ui_panel_queue = queue


def log_event(*args, level="INFO", from_ui=False, **kwargs):
    """
    A custom print function that writes to the log file, the standard
    logging module, and sends a structured log object to the UI queue.
    """
    global log_file_stream
    
    # Check if the last argument is a valid log level (handling common misuse of positional args)
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if args and isinstance(args[-1], str) and args[-1].upper() in valid_levels:
        # If the user passed the level as the last positional argument
        level = args[-1]
        args = args[:-1]

    message = " ".join(map(str, args))

    # Also write to the Python logger with the specified level
    level_upper = level.upper()
    if level_upper == "INFO":
        logging.info(message)
    elif level_upper == "WARNING":
        logging.warning(message)
    elif level_upper == "ERROR":
        logging.error(message)
    elif level_upper == "CRITICAL":
        logging.critical(message)
    elif level_upper == "DEBUG":
        logging.debug(message)
    else:
        logging.info(message)  # Default to INFO

    # If the UI queue is configured and this log didn't come from the UI,
    # create a structured log object and send it to the display.
    if ui_panel_queue is not None and not from_ui:
        try:
            log_object = {
                "type": "log_message",
                "timestamp": time.time(),
                "level": level.upper(),
                "message": message,
            }
            ui_panel_queue.put(log_object)
        except Exception as e:
            # If this fails, we can't display it, but we should log the failure.
            logging.error(f"Failed to queue log event for UI display: {e}")


class AnsiStrippingTee(object):
    """
    A thread-safe, file-like object that redirects stderr.
    It writes to the original stderr and to our log file, stripping ANSI codes.
    This is now primarily for capturing external library errors.
    """
    def __init__(self, stderr_stream):
        self.stderr_stream = stderr_stream # The original sys.stderr
        self.ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            # Write to the original stderr (for visibility in terminal)
            try:
                self.stderr_stream.write(data)
                self.stderr_stream.flush()
            except (IOError, ValueError):
                pass

            # Also write the stripped data to our central log_event function
            clean_data = self.ansi_escape.sub('', data)
            log_event(f"[STDERR] {clean_data.strip()}")

    def flush(self):
        with self.lock:
            try:
                self.stderr_stream.flush()
            except (IOError, ValueError):
                pass

    def isatty(self):
        # This helps libraries like 'rich' correctly render to stderr if needed.
        return hasattr(self.stderr_stream, 'isatty') and self.stderr_stream.isatty()


def set_log_level(level="INFO"):
    """
    Sets the logging level dynamically.
    
    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    level_upper = level.upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(level_upper, logging.INFO)
    logging.getLogger().setLevel(log_level)
    log_event(f"Log level set to {level_upper}", level="INFO")


def setup_global_logging(version_name='unknown', verbose=False):
    """
    Configures logging.
    - The `logging` module writes formatted logs to love.log.
    - `log_file_stream` provides a raw file handle to love.log for the custom `log_event`.
    - `sys.stderr` is redirected to our Tee to capture errors from external libraries.
    - `sys.stdout` is NOT redirected, so `rich.Console` can print UI panels directly.
    
    Args:
        version_name: Version identifier for the log
        verbose: If True, sets log level to DEBUG. Can also be set via LOVE_VERBOSE env var.
    """
    global log_file_stream
    
    # Check environment variable for verbose mode
    if os.getenv('LOVE_VERBOSE', '').lower() in ('1', 'true', 'yes'):
        verbose = True
    
    # Determine log level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # 1. Configure Python's logging module to write to the file.
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=LOG_FILE,
        filemode='a',
        force=True # Override any existing handlers
    )

    # 2. Open a raw file stream to the same log file for our custom print.
    # This captures unformatted text and stderr.
    if log_file_stream is None:
        log_file_stream = open(LOG_FILE, 'a')

    # 3. Redirect ONLY stderr to our custom Tee.
    # This is crucial for capturing errors from subprocesses or libraries (e.g., llama.cpp)
    # without interfering with our clean stdout for UI panels.
    original_stderr = sys.stderr
    sys.stderr = AnsiStrippingTee(original_stderr)

    # 4. Log the startup message using both methods.
    log_level_name = "DEBUG" if verbose else "INFO"
    startup_message = f"--- L.O.V.E. Version '{version_name}' session started (Log Level: {log_level_name}) ---"
    logging.info(startup_message)

    # We no longer print the startup message to stdout, as it's not a UI panel.
    # The console object will handle all direct user-facing output.


class ScopedDiagnosticLogger:
    """
    A context manager for logging the start, end, and exceptions of a code block.
    """
    def __init__(self, name, log_level=logging.DEBUG):
        self.name = name
        self.log_level_name = logging.getLevelName(log_level)

    def __enter__(self):
        log_event(f"Initiating: {self.name}", level=self.log_level_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            log_event(f"Completion: {self.name} - Failed with exception", level="ERROR")
            log_event(f"Exception in {self.name}: {exc_type.__name__}: {exc_val}", level="ERROR")
        else:
            log_event(f"Completion: {self.name} - Successful", level=self.log_level_name)

        # Return False to ensure any exception is re-raised.
        return False
