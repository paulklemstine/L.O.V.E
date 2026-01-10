#!/usr/bin/env python3
"""
SSH Web Terminal Server for L.O.V.E.
Serves a web page with xterm.js that connects via WebSocket.

DUAL MODE OPERATION:
- Observer Mode (default): All visitors see L.O.V.E. console output streaming in read-only mode
- Interactive Mode: After successful SSH authentication, full terminal access is granted

The console broadcast is received from love.py's simple_ui_renderer() via broadcast_to_observers().
"""

import asyncio
import os
import logging
import json
from aiohttp import web
import paramiko

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
WEB_PORT = int(os.environ.get('SSH_WEB_PORT', 8888))
SSH_HOST = os.environ.get('SSH_HOST', 'localhost')
SSH_PORT = int(os.environ.get('SSH_PORT', 22))
DEFAULT_USER = os.environ.get('SSH_USER', os.environ.get('USER', 'root'))

# --- BROADCAST INFRASTRUCTURE ---
# Set of WebSocket connections in observer mode (read-only)
observer_clients = set()
# Lock for thread-safe access to observer_clients
observer_lock = asyncio.Lock()
# Event loop reference for cross-thread broadcasting
_event_loop = None

def set_event_loop(loop):
    """Set the event loop for cross-thread broadcasting."""
    global _event_loop
    _event_loop = loop

def broadcast_to_observers(json_data: str):
    """
    Broadcast console output to all observer clients.
    Called from love.py's simple_ui_renderer() to relay console output.
    This is thread-safe and can be called from any thread.
    """
    global _event_loop
    if not _event_loop or not observer_clients:
        return
    
    try:
        asyncio.run_coroutine_threadsafe(
            _async_broadcast_to_observers(json_data),
            _event_loop
        )
    except Exception as e:
        logger.debug(f"Broadcast error (non-critical): {e}")

async def _async_broadcast_to_observers(json_data: str):
    """Async implementation of observer broadcast."""
    async with observer_lock:
        if not observer_clients:
            return
        
        # Wrap the data with a type indicator for the client
        message = json.dumps({
            "type": "console_output",
            "data": json.loads(json_data) if isinstance(json_data, str) else json_data
        })
        
        # Send to all observers, removing any that fail
        disconnected = set()
        for ws in observer_clients:
            try:
                await ws.send_str(message)
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            observer_clients.discard(ws)


# HTML template with observer mode and login form
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L.O.V.E. Console</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f23 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, sans-serif;
            display: flex;
            flex-direction: column;
        }
        header {
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 255, 136, 0.2);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.4em;
            font-weight: bold;
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
            font-size: 0.9em;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff4444;
            box-shadow: 0 0 10px currentColor;
            transition: all 0.3s ease;
        }
        .status-dot.connected { background: #00ff88; }
        .status-dot.observer { background: #00d4ff; }
        .status-dot.connecting { background: #ffaa00; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        /* Mode indicator badge */
        .mode-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .mode-badge.observer {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            border: 1px solid rgba(0, 212, 255, 0.4);
        }
        .mode-badge.interactive {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.4);
        }
        
        /* Login Overlay */
        #login-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .login-box {
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 12px;
            padding: 30px 40px;
            max-width: 400px;
            width: 100%;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), 0 0 60px rgba(0, 255, 136, 0.1);
        }
        .login-box h2 {
            color: #00ff88;
            margin-bottom: 20px;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            color: #888;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .form-group input {
            width: 100%;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 255, 136, 0.2);
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #00ff88;
        }
        .login-btn, .take-control-btn, .cancel-btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        .login-btn {
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            color: #000;
        }
        .take-control-btn {
            background: transparent;
            border: 1px solid rgba(0, 255, 136, 0.5);
            color: #00ff88;
        }
        .cancel-btn {
            background: transparent;
            border: 1px solid rgba(255, 100, 100, 0.5);
            color: #ff6464;
        }
        .login-btn:hover, .take-control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4);
        }
        .login-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .error-msg {
            color: #ff5555;
            text-align: center;
            margin-top: 15px;
            font-size: 0.9em;
            display: none;
        }
        
        /* Console/Terminal Container */
        #console-container {
            flex: 1;
            padding: 15px;
            display: flex;
            flex-direction: column;
        }
        #console, #terminal {
            flex: 1;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), 0 0 60px rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.15);
        }
        #console {
            background: #0a0a0a;
            padding: 15px;
            font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
            font-size: 13px;
            color: #e0e0e0;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        #terminal {
            display: none;
        }
        .xterm { height: 100%; padding: 10px; }
        .info-bar {
            background: rgba(0, 0, 0, 0.4);
            padding: 8px 15px;
            font-size: 0.8em;
            color: #666;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        /* Console output styling */
        .console-panel {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid #00ff88;
            background: rgba(0, 255, 136, 0.05);
        }
        .console-panel.error {
            border-left-color: #ff5555;
            background: rgba(255, 85, 85, 0.05);
        }
        .console-panel.warning {
            border-left-color: #ffaa00;
            background: rgba(255, 170, 0, 0.05);
        }
        .console-panel .title {
            color: #00ff88;
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">🖥️ L.O.V.E. Console</div>
        <div class="status">
            <div id="mode-badge" class="mode-badge observer">Observer</div>
            <div id="status-dot" class="status-dot connecting"></div>
            <span id="status-text">Connecting...</span>
        </div>
    </header>
    
    <!-- Console Output (Observer Mode) -->
    <div id="console-container">
        <div id="console"></div>
        <div id="terminal"></div>
    </div>
    
    <div class="info-bar">
        <span id="mode-info">Read-only mode - Click "Take Control" to authenticate</span>
        <button class="take-control-btn" id="take-control-btn" onclick="showLoginOverlay()">🔑 Take Control</button>
    </div>
    
    <!-- Login Overlay -->
    <div id="login-overlay">
        <div class="login-box">
            <h2>SSH Authentication</h2>
            <p style="color: #888; text-align: center; margin-bottom: 20px; font-size: 0.9em;">
                Authenticate to take control of the terminal
            </p>
            <form id="login-form">
                <div class="form-group">
                    <label for="host">Host</label>
                    <input type="text" id="host" value="SSH_HOST_PLACEHOLDER" required>
                </div>
                <div class="form-group">
                    <label for="port">Port</label>
                    <input type="number" id="port" value="SSH_PORT_PLACEHOLDER" required>
                </div>
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" value="DEFAULT_USER_PLACEHOLDER" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" placeholder="Enter password" required autofocus>
                </div>
                <button type="submit" class="login-btn" id="connect-btn">Connect</button>
                <button type="button" class="cancel-btn" onclick="hideLoginOverlay()">Cancel</button>
                <div class="error-msg" id="error-msg"></div>
            </form>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-web-links@0.9.0/lib/xterm-addon-web-links.min.js"></script>
    <script>
        let terminal = null;
        let fitAddon = null;
        let ws = null;
        let isInteractiveMode = false;
        let messageBuffer = [];
        const MAX_CONSOLE_LINES = 500;

        function setStatus(status, text) {
            const dot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            dot.className = 'status-dot ' + status;
            statusText.textContent = text;
        }

        function setMode(mode) {
            const badge = document.getElementById('mode-badge');
            const modeInfo = document.getElementById('mode-info');
            const takeControlBtn = document.getElementById('take-control-btn');
            const consoleEl = document.getElementById('console');
            const terminalEl = document.getElementById('terminal');
            
            if (mode === 'interactive') {
                badge.className = 'mode-badge interactive';
                badge.textContent = 'Interactive';
                modeInfo.textContent = 'Full terminal access - Press Ctrl+Shift+V to paste';
                takeControlBtn.style.display = 'none';
                consoleEl.style.display = 'none';
                terminalEl.style.display = 'block';
                isInteractiveMode = true;
            } else {
                badge.className = 'mode-badge observer';
                badge.textContent = 'Observer';
                modeInfo.textContent = 'Read-only mode - Click "Take Control" to authenticate';
                takeControlBtn.style.display = 'block';
                consoleEl.style.display = 'block';
                terminalEl.style.display = 'none';
                isInteractiveMode = false;
            }
        }

        function showError(msg) {
            const errorEl = document.getElementById('error-msg');
            errorEl.textContent = msg;
            errorEl.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error-msg').style.display = 'none';
        }

        function showLoginOverlay() {
            document.getElementById('login-overlay').style.display = 'flex';
            document.getElementById('password').focus();
        }

        function hideLoginOverlay() {
            document.getElementById('login-overlay').style.display = 'none';
            hideError();
        }

        function appendConsoleOutput(data) {
            const consoleEl = document.getElementById('console');
            
            // Handle panel format from love.py
            if (data.panel_type && data.content) {
                const panel = document.createElement('div');
                panel.className = 'console-panel';
                if (data.panel_type === 'error' || data.panel_type === 'api_error') {
                    panel.className += ' error';
                } else if (data.panel_type === 'warning') {
                    panel.className += ' warning';
                }
                
                if (data.title) {
                    const title = document.createElement('div');
                    title.className = 'title';
                    title.textContent = data.title;
                    panel.appendChild(title);
                }
                
                const content = document.createElement('div');
                content.textContent = data.content;
                panel.appendChild(content);
                
                consoleEl.appendChild(panel);
            } else {
                // Plain text output
                const line = document.createElement('div');
                line.textContent = typeof data === 'string' ? data : JSON.stringify(data);
                consoleEl.appendChild(line);
            }
            
            // Limit console lines to prevent memory issues
            while (consoleEl.children.length > MAX_CONSOLE_LINES) {
                consoleEl.removeChild(consoleEl.firstChild);
            }
            
            // Auto-scroll to bottom
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        function initTerminal() {
            terminal = new Terminal({
                cursorBlink: true,
                cursorStyle: 'block',
                fontSize: 14,
                fontFamily: '"Cascadia Code", "Fira Code", "JetBrains Mono", Consolas, monospace',
                theme: {
                    background: '#0a0a0a',
                    foreground: '#e0e0e0',
                    cursor: '#00ff88',
                    cursorAccent: '#000000',
                    selection: 'rgba(0, 255, 136, 0.3)',
                    black: '#000000',
                    red: '#ff5555',
                    green: '#00ff88',
                    yellow: '#ffaa00',
                    blue: '#00aaff',
                    magenta: '#ff55ff',
                    cyan: '#00d4ff',
                    white: '#e0e0e0',
                    brightBlack: '#555555',
                    brightRed: '#ff7777',
                    brightGreen: '#55ff99',
                    brightYellow: '#ffcc55',
                    brightBlue: '#55bbff',
                    brightMagenta: '#ff77ff',
                    brightCyan: '#55eeff',
                    brightWhite: '#ffffff'
                },
                allowProposedApi: true
            });

            fitAddon = new FitAddon.FitAddon();
            const webLinksAddon = new WebLinksAddon.WebLinksAddon();
            
            terminal.loadAddon(fitAddon);
            terminal.loadAddon(webLinksAddon);
            terminal.open(document.getElementById('terminal'));
            fitAddon.fit();

            window.addEventListener('resize', () => fitAddon.fit());

            terminal.onData((data) => {
                if (ws && ws.readyState === WebSocket.OPEN && isInteractiveMode) {
                    ws.send(JSON.stringify({ type: 'input', data: data }));
                }
            });

            terminal.onResize(({ cols, rows }) => {
                if (ws && ws.readyState === WebSocket.OPEN && isInteractiveMode) {
                    ws.send(JSON.stringify({ type: 'resize', cols, rows }));
                }
            });
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            setStatus('connecting', 'Connecting...');
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                setStatus('observer', 'Connected (Observer)');
                setMode('observer');
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    
                    if (msg.type === 'console_output') {
                        // Console broadcast for observers
                        appendConsoleOutput(msg.data);
                    } else if (msg.type === 'auth_success') {
                        // Authenticated - switch to interactive mode
                        hideLoginOverlay();
                        setMode('interactive');
                        setStatus('connected', 'Connected (Interactive)');
                        
                        if (!terminal) initTerminal();
                        terminal.focus();
                        
                        // Send terminal size
                        setTimeout(() => {
                            fitAddon.fit();
                            ws.send(JSON.stringify({ type: 'resize', cols: terminal.cols, rows: terminal.rows }));
                        }, 100);
                    } else if (msg.type === 'auth_failure') {
                        showError(msg.message || 'Authentication failed');
                        document.getElementById('connect-btn').disabled = false;
                    } else if (msg.type === 'data' && isInteractiveMode) {
                        // Terminal data for interactive mode
                        if (terminal) terminal.write(msg.data);
                    }
                } catch (e) {
                    // Plain text data for terminal
                    if (terminal && isInteractiveMode) terminal.write(event.data);
                }
            };

            ws.onclose = () => {
                setStatus('', 'Disconnected');
                setMode('observer');
                if (terminal) {
                    terminal.writeln('\\r\\n\\x1b[31m[Connection closed]\\x1b[0m');
                }
                // Attempt to reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setStatus('', 'Connection Error');
            };
        }

        function authenticate() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                showError('Not connected to server');
                return;
            }
            
            const host = document.getElementById('host').value;
            const port = document.getElementById('port').value;
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            document.getElementById('connect-btn').disabled = true;
            hideError();
            setStatus('connecting', 'Authenticating...');
            
            ws.send(JSON.stringify({ 
                type: 'auth', 
                host: host,
                port: parseInt(port),
                username: username, 
                password: password 
            }));
        }

        document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            authenticate();
        });
        
        // Connect on page load
        connectWebSocket();
    </script>
</body>
</html>
"""


class SSHSession:
    """Manages an SSH connection for a WebSocket client."""
    
    def __init__(self):
        self.client = None
        self.channel = None
        self.transport = None
    
    def connect(self, host, port, username, password):
        """Establish SSH connection with password auth."""
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        logger.info(f"Attempting SSH connection to {username}@{host}:{port}")
        self.client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            timeout=10,
            look_for_keys=False,
            allow_agent=False
        )
        
        self.transport = self.client.get_transport()
        self.channel = self.transport.open_session()
        self.channel.get_pty(term='xterm-256color', width=80, height=24)
        self.channel.invoke_shell()
        self.channel.setblocking(False)
        
        logger.info(f"SSH connected to {username}@{host}:{port}")
        return True
    
    def resize(self, cols, rows):
        """Resize the PTY."""
        if self.channel:
            self.channel.resize_pty(width=cols, height=rows)
    
    def send(self, data):
        """Send data to SSH channel."""
        if self.channel:
            self.channel.send(data)
    
    def recv(self, size=4096):
        """Receive data from SSH channel (non-blocking)."""
        if self.channel and self.channel.recv_ready():
            return self.channel.recv(size).decode('utf-8', errors='replace')
        return None
    
    def is_active(self):
        """Check if connection is still active."""
        return self.transport and self.transport.is_active()
    
    def close(self):
        """Close the SSH connection."""
        if self.channel:
            self.channel.close()
        if self.client:
            self.client.close()
        logger.info("SSH session closed")


async def websocket_handler(request):
    """
    Handle WebSocket connections for the console terminal.
    
    DUAL MODE OPERATION:
    - Starts in Observer mode: receives console broadcasts, read-only
    - Can upgrade to Interactive mode: after SSH authentication, full terminal access
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info(f"New WebSocket connection from {request.remote}")
    
    # Add to observer clients immediately (read-only mode)
    async with observer_lock:
        observer_clients.add(ws)
    
    ssh = None
    authenticated = False
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data.get('type') == 'auth' and not authenticated:
                        # Handle authentication - attempt to switch to interactive mode
                        host = data.get('host', SSH_HOST)
                        port = data.get('port', SSH_PORT)
                        username = data.get('username', DEFAULT_USER)
                        password = data.get('password', '')
                        
                        ssh = SSHSession()
                        try:
                            ssh.connect(host, port, username, password)
                            authenticated = True
                            
                            # Remove from observer clients - now in interactive mode
                            async with observer_lock:
                                observer_clients.discard(ws)
                            
                            await ws.send_str(json.dumps({'type': 'auth_success'}))
                            
                            # Start reading from SSH
                            asyncio.create_task(ssh_reader(ws, ssh))
                            
                        except Exception as e:
                            logger.error(f"SSH auth failed: {e}")
                            await ws.send_str(json.dumps({
                                'type': 'auth_failure',
                                'message': str(e)
                            }))
                    
                    elif authenticated and ssh:
                        if data.get('type') == 'input':
                            ssh.send(data.get('data', ''))
                        elif data.get('type') == 'resize':
                            cols = data.get('cols', 80)
                            rows = data.get('rows', 24)
                            ssh.resize(cols, rows)
                            
                except json.JSONDecodeError:
                    if authenticated and ssh:
                        ssh.send(msg.data)
                        
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break
    finally:
        # Clean up
        async with observer_lock:
            observer_clients.discard(ws)
        if ssh:
            ssh.close()
    
    logger.info("WebSocket connection closed")
    return ws


async def ssh_reader(ws, ssh):
    """Background task to read from SSH and send to WebSocket."""
    while not ws.closed and ssh.is_active():
        try:
            data = ssh.recv()
            if data:
                await ws.send_str(json.dumps({'type': 'data', 'data': data}))
            else:
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"SSH read error: {e}")
            break


async def index_handler(request):
    """Serve the main HTML page with default values filled in."""
    html = HTML_TEMPLATE.replace('SSH_HOST_PLACEHOLDER', SSH_HOST)
    html = html.replace('SSH_PORT_PLACEHOLDER', str(SSH_PORT))
    html = html.replace('DEFAULT_USER_PLACEHOLDER', DEFAULT_USER)
    return web.Response(text=html, content_type='text/html')


async def health_handler(request):
    """Health check endpoint."""
    return web.json_response({
        'status': 'ok', 
        'ssh_host': SSH_HOST, 
        'ssh_port': SSH_PORT,
        'observer_count': len(observer_clients)
    })


def create_app():
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/health', health_handler)
    return app


async def run_server():
    """Run the SSH web terminal server with event loop reference."""
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', WEB_PORT)
    await site.start()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           L.O.V.E. Console Viewer (SSH Web Terminal)         ║
╠══════════════════════════════════════════════════════════════╣
║  Web UI:    http://localhost:{WEB_PORT:<5}                          ║
║  Default SSH: {DEFAULT_USER}@{SSH_HOST}:{SSH_PORT:<5}                         ║
║                                                              ║
║  MODES:                                                      ║
║    • Observer: View console output (no login required)       ║
║    • Interactive: Full SSH access (requires authentication)  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Keep the server running
    await asyncio.Event().wait()


def main():
    """Start the SSH web terminal server."""
    asyncio.run(run_server())


if __name__ == '__main__':
    main()
