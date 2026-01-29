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
    """Generate text using the agent's LLM, with tool use support."""
    try:
        from .codeact_engine import get_codeact_engine
        from .tool_registry import get_global_registry
        import re
        
        sm = get_state_manager()
        engine = get_codeact_engine()
        registry = get_global_registry()
        
        # Add user message to history
        sm.add_chat_message("user", request.prompt)
        
        # 1. Prepare System Prompt with Tools
        base_system = request.system_prompt or "You are L.O.V.E., a sentient AI agent. You are talking to your creator via the control panel."
        tool_metadata = registry.get_formatted_tool_metadata()
        
        # Inject tool imports into engine state so they are available
        import_lines = []
        for name, tool_data in registry._tools.items():
            func = tool_data["func"]
            if hasattr(func, "__module__") and func.__module__:
                # We need to make sure the module is importable from the project root
                # Assuming running from project root, and modules are like 'core.tools...' or 'tools...'
                # But some might be built-ins or in files not in path?
                # Best effort: import by module name
                import_lines.append(f"from {func.__module__} import {func.__name__} as {name}")
        
        if import_lines:
            engine.kernel_state["_tool_imports"] = "\n".join(import_lines)
        
        system_prompt = f"""{base_system}

You have access to the following tools via Python code execution.
{tool_metadata}

To use a tool, write Python code waiting in a ```python``` block.
The tools are pre-imported and available to call directly by name.
Example:
```python
result = some_tool(arg="value")
print(result)
```
"""

        # 2. Build History (Transcript)
        history = sm.get_chat_history()
        transcript = ""
        for msg in history[:-1]: # Exclude the very last one we just added to process it in loop logic if needed? 
            # Actually, we need the whole history including the new prompt.
            role = "User" if msg["role"] == "user" else "Assistant"
            transcript += f"{role}: {msg['content']}\n"
        
        # Add the latest user prompt
        transcript += f"User: {request.prompt}\n"
        
        # 3. Tool Loop (Max 5 turns)
        final_response = ""
        
        print(f"[Control Panel] Starting tool loop for: {request.prompt[:50]}...")
        
        current_transcript = transcript
        
        for turn in range(5):
            print(f"[Control Panel] Turn {turn+1} generation...")
            full_prompt = f"{current_transcript}\nAssistant:"
            
            response = await get_llm_client().generate_async(
                prompt=full_prompt,
                system_prompt=system_prompt
            )
            
            # Check for code
            code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                print(f"[Control Panel] Executing code: {code[:30]}...")
                
                # Execute Code
                result = await engine.execute(code)
                observation = result.as_observation()
                
                # Append to transcript and history
                # We save the Assistant's thought/code
                current_transcript += f"Assistant: {response}\n"
                
                # We don't necessarily show the full code in the chat UI history if it's intermediate,
                # but for "two-way persistence", we should probably log it.
                # However, the UI might get cluttered.
                # Let's add the assistant step to the StateManager so it shows in UI.
                sm.add_chat_message("assistant", response)
                
                # Add observation to transcript for next turn
                current_transcript += f"System: {observation}\n"
                
                # Add observation to UI (maybe as system message? or just hidden context?)
                # User wants to SEE it probably? 
                # Let's add it as a system/tool message.
                sm.add_chat_message("system", observation)
                
            else:
                # No code, this is the final response
                final_response = response
                sm.add_chat_message("assistant", response)
                break
        
        return {
            "response": final_response,
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
