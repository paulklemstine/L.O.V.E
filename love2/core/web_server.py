from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import threading
from pathlib import Path
from .state_manager import get_state_manager

app = FastAPI(title="L.O.V.E. v2 Control Panel")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/status")
async def get_status():
    """Get current agent status."""
    return get_state_manager().get_snapshot()

@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """Get recent logs."""
    return get_state_manager().get_recent_logs(limit)

def find_available_port(start_port=8000, max_tries=10):
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    return None

def run_server(host="0.0.0.0", port=8000):
    """Run the web server."""
    # Note: We rely on the caller to find a free port, or we just try to run it.
    # But for the background thread, we want to find one.
    uvicorn.run(app, host=host, port=port, log_level="error")

def start_background_server(start_port=8000):
    """Start the server in a background thread on an available port."""
    port = find_available_port(start_port)
    if not port:
        print(f"[Error] Could not find available port between {start_port} and {start_port+10}")
        return None
    
    print(f"\n[Control Panel] 🌐 Web UI running at http://localhost:{port}")
    thread = threading.Thread(target=run_server, kwargs={"port": port}, daemon=True)
    thread.start()
    return thread
