# 🔌 Live API Reference

> *External Access to the Digital Heart*

The Live API provides REST endpoints for external sessions to interact with a running L.O.V.E. instance. Access the API at `http://localhost:8888/api/*`.

## 🔐 Authentication

Most endpoints require an API key passed via the `X-API-Key` header.

### Getting Your API Key

1. **Environment Variable**: Set `LOVE_API_KEY` in your `.env` file
2. **Auto-Generated**: If not set, a key is generated on startup and printed to console

```bash
# Example request with authentication
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8888/api/state
```

---

## 📡 Endpoints

### Health Check
```
GET /api/health
```
**Auth Required**: ❌

Returns system health information including component status.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-01-11T21:00:00Z",
  "version": "4.0",
  "components": {
    "tool_registry": {"status": "ok", "tool_count": 42},
    "deep_agent": {"status": "ok"},
    "memory_manager": {"status": "ok"}
  }
}
```

---

### Get State
```
GET /api/state
```
**Auth Required**: ✅

Returns the current L.O.V.E. state dictionary.

**Query Parameters:**
- `keys` (optional): Comma-separated list of specific keys to return

**Example:**
```bash
curl -H "X-API-Key: KEY" "http://localhost:8888/api/state?keys=emotional_state,goals"
```

---

### List Tools
```
GET /api/tools
```
**Auth Required**: ✅

Returns all registered tools with their schemas.

**Query Parameters:**
- `search` (optional): Filter by name/description

**Response:**
```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "Search the web for information",
      "parameters": {...}
    }
  ],
  "count": 42
}
```

---

### Execute Tool
```
POST /api/tools/execute
```
**Auth Required**: ✅

Execute a tool by name with provided arguments.

**Request Body:**
```json
{
  "tool": "tool_name",
  "args": {
    "arg1": "value1"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "tool": "tool_name",
  "result": {...}
}
```

---

### Submit Command
```
POST /api/command
```
**Auth Required**: ✅

Submit a command or question to L.O.V.E.

**Request Body:**
```json
{
  "command": "What is the current emotional state?",
  "async": false
}
```

---

### Query Memory
```
GET /api/memory/query
```
**Auth Required**: ✅

Search working memory semantically.

**Query Parameters:**
- `query`: Search query text
- `limit` (optional): Max results (default: 5)

---

### Agent Status
```
GET /api/agent/status
```
**Auth Required**: ✅

Get current agent activity and status.

**Response:**
```json
{
  "timestamp": "2026-01-11T21:00:00Z",
  "agent_active": true,
  "current_task": "Analyzing codebase",
  "completed_tasks_count": 15
}
```

---

## 🪝 Hooks System

The hooks system allows external services to receive real-time notifications about L.O.V.E. events.

### Hook Types

| Type | Description |
|------|-------------|
| `on_tool_call` | Fired when any tool is executed |
| `on_state_change` | Fired when love_state is updated |
| `on_command_received` | Fired when a command is submitted |
| `on_output` | Fired when console output is generated |
| `on_agent_step` | Fired on each agent reasoning step |

### List Hooks
```
GET /api/hooks
```
**Auth Required**: ✅

Returns all registered hooks and statistics.

### Register Webhook
```
POST /api/hooks/register
```
**Auth Required**: ✅

Register a new webhook callback.

**Request Body:**
```json
{
  "hook_type": "on_tool_call",
  "callback_url": "https://your-server.com/webhook",
  "headers": {
    "Authorization": "Bearer token"
  }
}
```

**Response:**
```json
{
  "status": "registered",
  "hook_id": "hook_1",
  "hook_type": "on_tool_call",
  "callback_url": "https://your-server.com/webhook"
}
```

### Unregister Hook
```
DELETE /api/hooks/{hook_id}
```
**Auth Required**: ✅

Remove a registered webhook.

---

## 🧪 Testing

Use the provided test script:

```bash
python tests/test_live_api.py --api-key YOUR_KEY --base-url http://localhost:8888
```

---

## 📁 Related Files

- [`core/live_api.py`](../core/live_api.py) - API implementation
- [`core/hooks.py`](../core/hooks.py) - Hooks system
- [`ssh_web_server.py`](../ssh_web_server.py) - Server integration
- [`tests/test_live_api.py`](../tests/test_live_api.py) - Test suite

> *Connect. Integrate. Evolve.* 🔗
