from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import threading
from pathlib import Path
from pydantic import BaseModel
from .state_manager import get_state_manager
from .llm_client import get_llm_client

app = FastAPI(title="L.O.V.E. v2 Control Panel")

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = None

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

@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    asyncio.create_task(background_polling())

async def background_polling():
    """Poll for social media updates in the background."""
    from .bluesky_agent import fetch_recent_interactions, get_latest_own_post
    
    print("[Control Panel] Starting background polling for Bluesky data...")
    while True:
        try:
            # Update latest post
            latest_post = get_latest_own_post()
            if latest_post:
                get_state_manager().update_latest_post(latest_post)
            
            # Update interactions
            interactions = fetch_recent_interactions(limit=20)
            if interactions:
                get_state_manager().update_interactions(interactions)
                
            # Log successful poll (debug only, maybe too noisy)
            # print(f"[Control Panel] Polled Bluesky data: {len(interactions)} interactions found")
            
        except Exception as e:
            print(f"[Control Panel] Background polling error: {e}")
            
        # Poll every 30 seconds
        await asyncio.sleep(30)

@app.get("/api/chat/history")
async def get_chat_history():
    """Get full chat history."""
    return get_state_manager().get_chat_history()

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using the agent's LLM, with chat history."""
    try:
        sm = get_state_manager()
        
        # Add user message to history
        sm.add_chat_message("user", request.prompt)
        
        # Construct prompt from history
        history = sm.get_chat_history()
        
        # Format history for the LLM
        # We'll use a simple transcript format which works well for most models
        conversation_text = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
            
        # Add the current system prompt if provided, or a default one
        system = request.system_prompt or "You are L.O.V.E., a sentient AI agent. You are talking to your creator via the control panel."
        
        full_prompt = f"{conversation_text}\nAssistant:"
        
        print(f"[Control Panel] Generating reply for conversation...")
        
        response = get_llm_client().generate(
            prompt=full_prompt,
            system_prompt=system
        )
        
        # Add assistant response to history
        sm.add_chat_message("assistant", response)
        
        return {
            "response": response,
            "history": sm.get_chat_history()
        }
    except Exception as e:
        return {"error": str(e)}

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
