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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 3. Web UI Handler
    web_handler = WebUIHandler()
    web_handler.setFormatter(formatter)
    root_logger.addHandler(web_handler)
    
    # Redirect stdout to capture print statements
    # WARNING: Do NOT redirect stderr, as it causes infinite recursion if the logger itself fails
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    # sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
    
    logging.info("Logging initialized. Writing to %s", LOG_FILE)

def log_event(message: str, level: str = "INFO"):
    """
    Helper function for v1 compatibility.
    Logs an event with the specified level.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.log(lvl, message)

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
