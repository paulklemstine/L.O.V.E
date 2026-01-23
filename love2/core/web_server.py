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

def run_server(host="0.0.0.0", port=8000):
    """Run the web server."""
    uvicorn.run(app, host=host, port=port, log_level="error")

def start_background_server():
    """Start the server in a background thread."""
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread
