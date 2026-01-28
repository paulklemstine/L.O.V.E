import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .state_manager import get_state_manager

# Create logs directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "love2.log"

class WebUIHandler(logging.Handler):
    """Custom handler to push logs to StateManager."""
    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            get_state_manager().add_log(level, msg, record.name)
        except Exception:
            self.handleError(record)

def setup_logging(verbose: bool = False):
    """Configure logging for L.O.V.E. v2."""
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if not verbose else logging.DEBUG)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. File Handler (Rotating)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5*1024*1024, # 5MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 2. Console Handler
    # Check if sys.stdout is already our wrapper (from a previous run)
    raw_stdout = sys.stdout
    if isinstance(raw_stdout, StreamToLogger):
         # If already wrapped, we won't wrap it again, but we need the underlying stream for the handler
         # Since we don't store it in StreamToLogger (my bad), we'll try sys.__stdout__ as fallback
         # But safer: avoid re-wrapping at the end of function
         # Ideally we should store the original stream
         pass
    
    # Use the raw stream for the handler to prevent loops
    # If sys.stdout is already wrapped, this might be risky unless we unwrap it.
    # Let's perform a check.
    
    if hasattr(sys.stdout, 'is_pseudo'): 
        # Detect simple wrapper if we tag it, but for now:
        pass

    # Better Strategy:
    # 1. Capture current stdout
    current_stdout = sys.stdout
    
    # 2. If it's ALREADY a StreamToLogger, DO NOT re-wrap, and use its logger for output? 
    # No, we want to reset the handlers.
    
    # If it is our wrapper, let's try to unwrap or fallback to __stdout__
    target_stream = current_stdout
    if isinstance(target_stream, StreamToLogger):
        # Fallback to __stdout__ if available, otherwise we are stuck
        target_stream = getattr(sys, '__stdout__', target_stream)
        
    console_handler = logging.StreamHandler(target_stream)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 3. Web UI Handler
    web_handler = WebUIHandler()
    web_handler.setFormatter(formatter)
    root_logger.addHandler(web_handler)
    
    # Redirect stdout to capture print statements
    # ONLY wrap if not already wrapped
    if not isinstance(sys.stdout, StreamToLogger):
        sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    
    logging.info("Logging initialized. Writing to %s", LOG_FILE)

def log_event(message: str, level: str = "INFO"):
    """
    Helper function for v1 compatibility.
    Logs an event with the specified level.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.log(lvl, message)


def get_logger(name: str):
    """
    Get a logger instance by name.
    
    Standard Python logging interface for introspection module.
    """
    return logging.getLogger(name)

class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        return False
