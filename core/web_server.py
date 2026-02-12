from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import threading
import json
from pathlib import Path
from pydantic import BaseModel
from .state_manager import get_state_manager
from .llm_client import get_llm_client
from .pi_rpc_bridge import get_pi_bridge

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

@app.get("/interactive")
async def read_interactive():
    return FileResponse(STATIC_DIR / "interactive.html")

@app.websocket("/ws/pi")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    bridge = get_pi_bridge()
    
    # Callback to forward agent events to UI
    async def on_agent_event(event_data):
        try:
            await websocket.send_json(event_data)
        except Exception:
            pass # Socket likely closed

    bridge.set_callback(on_agent_event, callback_id="web_ui")
    
    # Ensure bridge is started
    if not bridge.running:
        await bridge.start()
    
    try:
        while True:
            data = await websocket.receive_text()
            # User input from UI
            # We expect a simple string prompt for now, or JSON command
            # If it's a JSON string, try to parse it
            try:
                cmd = json.loads(data)
                # If command object, send as is
                if isinstance(cmd, dict) and "type" in cmd:
                    await bridge.send_command_json(cmd)
                else:
                    # Treat as prompt
                    await bridge.send_prompt(str(data))
            except json.JSONDecodeError:
                # Treat raw text as prompt
                 await bridge.send_prompt(data)
                 
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
        # Optional: stop agent if no clients? 
        # await bridge.stop() 
    except Exception as e:
        print(f"[WebSocket] Error: {e}")

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
    """
    Handle user messages from the control panel.
    Routes directly through Pi Agent for real-time interaction.
    """
    try:
        sm = get_state_manager()
        bridge = get_pi_bridge()
        
        # Add user message to history
        sm.add_chat_message("user", request.prompt)
        
        # Ensure Pi Agent bridge is running
        if not bridge.running:
            await bridge.start()
        
        # Collect Pi Agent's response
        response_text = []
        response_complete = asyncio.Event()
        import uuid
        callback_id = f"chat_{uuid.uuid4().hex[:8]}"
        
        async def handle_event(event: dict):
            event_type = event.get("type", "")
            if event_type == "message_update":
                update_type = event.get("update_type", "")
                if update_type == "text_delta":
                    text = event.get("text", "")
                    if text:
                        response_text.append(text)
            elif event_type in ("result", "message_complete", "error"):
                response_complete.set()
        
        bridge.set_callback(handle_event, callback_id=callback_id)
        
        try:
            await bridge.send_prompt(request.prompt)
            await asyncio.wait_for(response_complete.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            response_text.append("[Response timed out]")
        finally:
            bridge.remove_callback(callback_id)
        
        full_response = "".join(response_text).strip()
        if not full_response:
            full_response = "I received your message but had no response."
        
        # Add L.O.V.E.'s response to chat history
        sm.add_chat_message("assistant", full_response)
        
        return {
            "response": full_response,
            "history": sm.get_chat_history()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
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
