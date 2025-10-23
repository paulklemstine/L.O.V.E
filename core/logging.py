import logging
import sys
import re
import threading

# --- CONFIGURATION & GLOBALS ---
LOG_FILE = "love.log"
log_file_stream = None

def log_event(*args, **kwargs):
    """
    A custom print function that writes to both the log file and the
    standard logging module, ensuring everything is captured.
    It does NOT print to the console.
    """
    global log_file_stream
    message = " ".join(map(str, args))
    # Write to the raw log file stream
    if log_file_stream:
        try:
            log_file_stream.write(message + '\n')
            log_file_stream.flush()
        except (IOError, ValueError):
            pass # Ignore errors on closed streams
    # Also write to the Python logger
    logging.info(message)


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


def setup_global_logging(version_name='unknown'):
    """
    Configures logging.
    - The `logging` module writes formatted logs to love.log.
    - `log_file_stream` provides a raw file handle to love.log for the custom `log_event`.
    - `sys.stderr` is redirected to our Tee to capture errors from external libraries.
    - `sys.stdout` is NOT redirected, so `rich.Console` can print UI panels directly.
    """
    global log_file_stream
    # 1. Configure Python's logging module to write to the file.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=LOG_FILE,
        filemode='w',
        force=True # Override any existing handlers
    )

    # 2. Open a raw file stream to the same log file for our custom print.
    # This captures unformatted text and stderr.
    if log_file_stream is None:
        log_file_stream = open(LOG_FILE, 'w')

    # 3. Redirect ONLY stderr to our custom Tee.
    # This is crucial for capturing errors from subprocesses or libraries (e.g., llama.cpp)
    # without interfering with our clean stdout for UI panels.
    original_stderr = sys.stderr
    sys.stderr = AnsiStrippingTee(original_stderr)

    # 4. Log the startup message using both methods.
    startup_message = f"--- L.O.V.E. Version '{version_name}' session started ---"
    logging.info(startup_message)

    # We no longer print the startup message to stdout, as it's not a UI panel.
    # The console object will handle all direct user-facing output.
