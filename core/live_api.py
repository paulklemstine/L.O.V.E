"""
Live API Module for L.O.V.E. External Access.

This module provides REST API endpoints for external sessions to interact
with a running L.O.V.E. instance. Integrates with the SSH web server.

Endpoints:
    GET  /api/state          - Get current L.O.V.E. state
    GET  /api/tools          - List all registered tools
    POST /api/tools/execute  - Execute a tool by name
    POST /api/command        - Submit a command to L.O.V.E.
    GET  /api/memory/query   - Query working memory
    GET  /api/health         - Extended health check
    GET  /api/agent/status   - Current agent activity
    GET  /api/hooks          - List registered hooks
    POST /api/hooks/register - Register a new hook

Security:
    All endpoints except /api/health require X-API-Key header.
    Key is read from LOVE_API_KEY env var or auto-generated.
"""

import os
import json
import secrets
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps

from aiohttp import web

logger = logging.getLogger(__name__)

# API Key management
_api_key: Optional[str] = None


def get_api_key() -> str:
    """
    Get or generate the API key.
    
    Priority:
    1. LOVE_API_KEY environment variable
    2. Auto-generated key (printed to console on first access)
    """
    global _api_key
    
    if _api_key is not None:
        return _api_key
    
    # Try environment variable first
    env_key = os.environ.get("LOVE_API_KEY")
    if env_key:
        _api_key = env_key
        logger.info("Using API key from LOVE_API_KEY environment variable")
    else:
        # Generate a new key
        _api_key = secrets.token_urlsafe(32)
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  L.O.V.E. Live API Key                       ║
╠══════════════════════════════════════════════════════════════╣
║  API Key: {_api_key[:20]}...{_api_key[-8:]:>8}  ║
║                                                              ║
║  Use header: X-API-Key: {_api_key[:20]}...       ║
║  Or set LOVE_API_KEY env var to persist                      ║
╚══════════════════════════════════════════════════════════════╝
        """)
        logger.info("Generated new API key (set LOVE_API_KEY to persist)")
    
    return _api_key


def require_api_key(handler):
    """Decorator to require API key authentication."""
    @wraps(handler)
    async def wrapper(request):
        provided_key = request.headers.get("X-API-Key")
        expected_key = get_api_key()
        
        if not provided_key:
            return web.json_response(
                {"error": "Missing X-API-Key header"},
                status=401
            )
        
        if provided_key != expected_key:
            return web.json_response(
                {"error": "Invalid API key"},
                status=403
            )
        
        return await handler(request)
    
    return wrapper


# =============================================================================
# API Handlers
# =============================================================================

async def api_health(request) -> web.Response:
    """
    Extended health check endpoint.
    
    GET /api/health
    
    Returns:
        System health info including uptime, memory, and component status.
    """
    try:
        import core.shared_state as shared_state
        import psutil
    except ImportError:
        psutil = None
    
    # Get basic health info
    health = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0",
        "api_version": "1.0"
    }
    
    # Add system metrics if psutil available
    if psutil:
        health["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        }
    
    # Check component status
    components = {}
    
    if hasattr(shared_state, 'tool_registry') and shared_state.tool_registry:
        components["tool_registry"] = {
            "status": "ok",
            "tool_count": len(shared_state.tool_registry)
        }
    
    if hasattr(shared_state, 'deep_agent_engine') and shared_state.deep_agent_engine:
        components["deep_agent"] = {"status": "ok"}
    
    if hasattr(shared_state, 'memory_manager') and shared_state.memory_manager:
        components["memory_manager"] = {"status": "ok"}
    
    health["components"] = components
    
    return web.json_response(health)


@require_api_key
async def api_get_state(request) -> web.Response:
    """
    Get current L.O.V.E. state.
    
    GET /api/state
    
    Query params:
        keys: Comma-separated list of specific keys to return
        
    Returns:
        Current state dictionary (filtered by keys if specified).
    """
    try:
        import core.shared_state as shared_state
    except ImportError:
        return web.json_response({"error": "shared_state not available"}, status=500)
    
    state = shared_state.love_state.copy() if shared_state.love_state else {}
    
    # Filter by requested keys
    keys_param = request.query.get("keys")
    if keys_param:
        requested_keys = [k.strip() for k in keys_param.split(",")]
        state = {k: v for k, v in state.items() if k in requested_keys}
    
    # Add some computed fields
    state["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "key_count": len(shared_state.love_state) if shared_state.love_state else 0
    }
    
    return web.json_response(state)


@require_api_key
async def api_list_tools(request) -> web.Response:
    """
    List all registered tools.
    
    GET /api/tools
    
    Query params:
        search: Filter tools by name/description containing this string
        
    Returns:
        List of tool schemas.
    """
    try:
        import core.shared_state as shared_state
    except ImportError:
        return web.json_response({"error": "shared_state not available"}, status=500)
    
    if not shared_state.tool_registry:
        return web.json_response({"tools": [], "count": 0})
    
    # Get all tool schemas
    tools = []
    for name in shared_state.tool_registry.list_tools():
        schema = shared_state.tool_registry.get_schema(name)
        if schema:
            tools.append({
                "name": name,
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {})
            })
    
    # Filter by search query
    search = request.query.get("search", "").lower()
    if search:
        tools = [
            t for t in tools
            if search in t["name"].lower() or search in t.get("description", "").lower()
        ]
    
    return web.json_response({
        "tools": tools,
        "count": len(tools)
    })


@require_api_key
async def api_execute_tool(request) -> web.Response:
    """
    Execute a tool by name.
    
    POST /api/tools/execute
    
    Body:
        {
            "tool": "tool_name",
            "args": {"arg1": "value1", ...}
        }
        
    Returns:
        Tool execution result.
    """
    try:
        import core.shared_state as shared_state
        from core.hooks import fire_tool_call
    except ImportError as e:
        return web.json_response({"error": f"Import error: {e}"}, status=500)
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    
    tool_name = data.get("tool")
    args = data.get("args", {})
    
    if not tool_name:
        return web.json_response({"error": "Missing 'tool' field"}, status=400)
    
    if not shared_state.tool_registry:
        return web.json_response({"error": "Tool registry not initialized"}, status=500)
    
    # Get the tool
    try:
        tool_func = shared_state.tool_registry.get_tool(tool_name)
    except KeyError as e:
        return web.json_response({"error": str(e)}, status=404)
    
    # Execute the tool
    try:
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**args)
        else:
            result = tool_func(**args)
        
        # Fire hook
        try:
            await fire_tool_call(tool_name, args, result)
        except Exception as hook_error:
            logger.warning(f"Hook fire failed: {hook_error}")
        
        # Serialize result
        if hasattr(result, "to_dict"):
            result_data = result.to_dict()
        elif hasattr(result, "__dict__"):
            result_data = result.__dict__
        else:
            result_data = str(result)
        
        return web.json_response({
            "status": "success",
            "tool": tool_name,
            "result": result_data
        })
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return web.json_response({
            "status": "error",
            "tool": tool_name,
            "error": str(e),
            "error_type": type(e).__name__
        }, status=500)


@require_api_key
async def api_submit_command(request) -> web.Response:
    """
    Submit a command/request to L.O.V.E.
    
    POST /api/command
    
    Body:
        {
            "command": "Your command or question here",
            "async": false  // If true, returns immediately with task ID
        }
        
    Returns:
        Command acknowledgment or result.
    """
    try:
        import core.shared_state as shared_state
        from core.hooks import fire_command
    except ImportError as e:
        return web.json_response({"error": f"Import error: {e}"}, status=500)
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    
    command = data.get("command")
    is_async = data.get("async", False)
    
    if not command:
        return web.json_response({"error": "Missing 'command' field"}, status=400)
    
    # Fire the command hook
    try:
        await fire_command(command, source="api")
    except Exception as e:
        logger.warning(f"Command hook failed: {e}")
    
    # Add to UI queue if available
    if shared_state.ui_panel_queue:
        shared_state.ui_panel_queue.put({
            "panel_type": "api_command",
            "title": "API Command",
            "content": command,
            "source": "live_api"
        })
    
    # TODO: Integrate with task manager for actual command execution
    # For now, acknowledge receipt
    
    return web.json_response({
        "status": "received",
        "command": command,
        "timestamp": datetime.now().isoformat(),
        "message": "Command queued for processing"
    })


@require_api_key
async def api_query_memory(request) -> web.Response:
    """
    Query working memory.
    
    GET /api/memory/query
    
    Query params:
        query: Semantic search query
        limit: Max results (default 5)
        
    Returns:
        Memory search results.
    """
    try:
        import core.shared_state as shared_state
    except ImportError:
        return web.json_response({"error": "shared_state not available"}, status=500)
    
    query = request.query.get("query", "")
    limit = int(request.query.get("limit", 5))
    
    if not shared_state.memory_manager:
        return web.json_response({
            "error": "Memory manager not initialized",
            "results": []
        }, status=500)
    
    try:
        # Try to search memory
        if hasattr(shared_state.memory_manager, 'search'):
            results = shared_state.memory_manager.search(query, limit=limit)
        elif hasattr(shared_state.memory_manager, 'query'):
            results = shared_state.memory_manager.query(query, limit=limit)
        else:
            results = []
        
        return web.json_response({
            "query": query,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        return web.json_response({
            "error": str(e),
            "results": []
        }, status=500)


@require_api_key
async def api_agent_status(request) -> web.Response:
    """
    Get current agent activity status.
    
    GET /api/agent/status
    
    Returns:
        Current agent state and activity.
    """
    try:
        import core.shared_state as shared_state
    except ImportError:
        return web.json_response({"error": "shared_state not available"}, status=500)
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "agent_active": False,
        "current_task": None,
        "completed_tasks_count": len(shared_state.completed_tasks)
    }
    
    if shared_state.deep_agent_engine:
        status["agent_active"] = True
        # Add more details if available
        if hasattr(shared_state.deep_agent_engine, 'current_goal'):
            status["current_goal"] = shared_state.deep_agent_engine.current_goal
    
    if shared_state.love_task_manager:
        if hasattr(shared_state.love_task_manager, 'current_task'):
            status["current_task"] = str(shared_state.love_task_manager.current_task)
    
    return web.json_response(status)


@require_api_key
async def api_list_hooks(request) -> web.Response:
    """
    List all registered hooks.
    
    GET /api/hooks
    
    Returns:
        All registered hooks by type.
    """
    try:
        from core.hooks import get_hooks_manager
    except ImportError:
        return web.json_response({"error": "Hooks module not available"}, status=500)
    
    manager = get_hooks_manager()
    hooks = manager.list_hooks()
    stats = manager.get_stats()
    
    return web.json_response({
        "hooks": hooks,
        "stats": stats
    })


@require_api_key
async def api_register_hook(request) -> web.Response:
    """
    Register a new webhook.
    
    POST /api/hooks/register
    
    Body:
        {
            "hook_type": "on_tool_call",  // See HookType enum
            "callback_url": "https://example.com/webhook",
            "headers": {"Authorization": "Bearer token"}  // Optional
        }
        
    Returns:
        Hook registration details.
    """
    try:
        from core.hooks import get_hooks_manager, HookType
    except ImportError:
        return web.json_response({"error": "Hooks module not available"}, status=500)
    
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    
    hook_type_str = data.get("hook_type")
    callback_url = data.get("callback_url")
    headers = data.get("headers", {})
    
    if not hook_type_str:
        return web.json_response({"error": "Missing 'hook_type' field"}, status=400)
    
    if not callback_url:
        return web.json_response({"error": "Missing 'callback_url' field"}, status=400)
    
    # Validate hook type
    try:
        hook_type = HookType(hook_type_str)
    except ValueError:
        valid_types = [ht.value for ht in HookType]
        return web.json_response({
            "error": f"Invalid hook_type. Valid types: {valid_types}"
        }, status=400)
    
    # Register the hook
    manager = get_hooks_manager()
    hook_id = manager.register_remote(hook_type, callback_url, headers)
    
    return web.json_response({
        "status": "registered",
        "hook_id": hook_id,
        "hook_type": hook_type_str,
        "callback_url": callback_url
    })


@require_api_key
async def api_unregister_hook(request) -> web.Response:
    """
    Unregister a webhook.
    
    DELETE /api/hooks/{hook_id}
    
    Returns:
        Unregistration confirmation.
    """
    try:
        from core.hooks import get_hooks_manager
    except ImportError:
        return web.json_response({"error": "Hooks module not available"}, status=500)
    
    hook_id = request.match_info.get("hook_id")
    
    if not hook_id:
        return web.json_response({"error": "Missing hook_id"}, status=400)
    
    manager = get_hooks_manager()
    removed = manager.unregister(hook_id)
    
    if removed:
        return web.json_response({
            "status": "unregistered",
            "hook_id": hook_id
        })
    else:
        return web.json_response({
            "error": f"Hook {hook_id} not found"
        }, status=404)


# =============================================================================
# Route Setup
# =============================================================================

def setup_live_api_routes(app: web.Application) -> None:
    """
    Add Live API routes to an aiohttp application.
    
    Args:
        app: The aiohttp Application to add routes to.
    """
    app.router.add_get("/api/health", api_health)
    app.router.add_get("/api/state", api_get_state)
    app.router.add_get("/api/tools", api_list_tools)
    app.router.add_post("/api/tools/execute", api_execute_tool)
    app.router.add_post("/api/command", api_submit_command)
    app.router.add_get("/api/memory/query", api_query_memory)
    app.router.add_get("/api/agent/status", api_agent_status)
    app.router.add_get("/api/hooks", api_list_hooks)
    app.router.add_post("/api/hooks/register", api_register_hook)
    app.router.add_delete("/api/hooks/{hook_id}", api_unregister_hook)
    
    logger.info("Live API routes configured")
    print("🔌 Live API endpoints available at /api/*")
