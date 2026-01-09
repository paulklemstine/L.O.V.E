#!/usr/bin/env python3
"""
SSH Web Terminal Server for L.O.V.E.
Serves a web page with xterm.js that connects via WebSocket to SSH.
Includes login form for password authentication.
"""

import asyncio
import os
import logging
from aiohttp import web
import paramiko

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
WEB_PORT = int(os.environ.get('SSH_WEB_PORT', 8888))
SSH_HOST = os.environ.get('SSH_HOST', 'localhost')
SSH_PORT = int(os.environ.get('SSH_PORT', 22))
DEFAULT_USER = os.environ.get('SSH_USER', os.environ.get('USER', 'root'))

# HTML template with login form and embedded xterm.js
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L.O.V.E. SSH Terminal</title>
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
        .status-dot.connecting { background: #ffaa00; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        /* Login Form */
        #login-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .login-box {
            background: rgba(0, 0, 0, 0.7);
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
        .login-btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            border: none;
            border-radius: 6px;
            color: #000;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .login-btn:hover {
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
        
        /* Terminal */
        #terminal-container {
            flex: 1;
            padding: 15px;
            display: none;
            flex-direction: column;
        }
        #terminal {
            flex: 1;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), 0 0 60px rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.15);
        }
        .xterm { height: 100%; padding: 10px; }
        .info-bar {
            background: rgba(0, 0, 0, 0.4);
            padding: 8px 15px;
            font-size: 0.8em;
            color: #666;
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">🖥️ L.O.V.E. SSH Terminal</div>
        <div class="status">
            <div id="status-dot" class="status-dot"></div>
            <span id="status-text">Not Connected</span>
        </div>
    </header>
    
    <!-- Login Form -->
    <div id="login-container">
        <div class="login-box">
            <h2>SSH Login</h2>
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
                <div class="error-msg" id="error-msg"></div>
            </form>
        </div>
    </div>
    
    <!-- Terminal -->
    <div id="terminal-container">
        <div id="terminal"></div>
    </div>
    <div class="info-bar" id="info-bar" style="display: none;">
        <span>Press Ctrl+Shift+V to paste | Ctrl+C to interrupt</span>
        <span id="connection-info"></span>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-web-links@0.9.0/lib/xterm-addon-web-links.min.js"></script>
    <script>
        let terminal = null;
        let fitAddon = null;
        let ws = null;

        function setStatus(status, text) {
            const dot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            dot.className = 'status-dot ' + status;
            statusText.textContent = text;
        }

        function showError(msg) {
            const errorEl = document.getElementById('error-msg');
            errorEl.textContent = msg;
            errorEl.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error-msg').style.display = 'none';
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
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'input', data: data }));
                }
            });

            terminal.onResize(({ cols, rows }) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'resize', cols, rows }));
                }
            });
        }

        function connect(host, port, username, password) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            setStatus('connecting', 'Connecting...');
            document.getElementById('connect-btn').disabled = true;
            hideError();

            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                // Send credentials
                ws.send(JSON.stringify({ 
                    type: 'auth', 
                    host: host,
                    port: parseInt(port),
                    username: username, 
                    password: password 
                }));
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'auth_success') {
                        // Show terminal, hide login
                        document.getElementById('login-container').style.display = 'none';
                        document.getElementById('terminal-container').style.display = 'flex';
                        document.getElementById('info-bar').style.display = 'flex';
                        
                        if (!terminal) initTerminal();
                        
                        setStatus('connected', 'Connected');
                        document.getElementById('connection-info').textContent = 
                            `Connected to ${username}@${host}:${port}`;
                        terminal.focus();
                        
                        // Send terminal size
                        setTimeout(() => {
                            fitAddon.fit();
                            ws.send(JSON.stringify({ type: 'resize', cols: terminal.cols, rows: terminal.rows }));
                        }, 100);
                    } else if (data.type === 'auth_failure') {
                        showError(data.message || 'Authentication failed');
                        setStatus('', 'Not Connected');
                        document.getElementById('connect-btn').disabled = false;
                        ws.close();
                    } else if (data.type === 'data') {
                        if (terminal) terminal.write(data.data);
                    }
                } catch (e) {
                    // Plain text data for terminal
                    if (terminal) terminal.write(event.data);
                }
            };

            ws.onclose = () => {
                setStatus('', 'Disconnected');
                document.getElementById('connect-btn').disabled = false;
                if (terminal) {
                    terminal.writeln('\\r\\n\\x1b[31m[Connection closed]\\x1b[0m');
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                showError('Connection error');
                document.getElementById('connect-btn').disabled = false;
            };
        }

        document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const host = document.getElementById('host').value;
            const port = document.getElementById('port').value;
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            connect(host, port, username, password);
        });
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
    """Handle WebSocket connections for SSH terminal."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info(f"New WebSocket connection from {request.remote}")
    
    ssh = None
    authenticated = False
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                import json
                try:
                    data = json.loads(msg.data)
                    
                    if data.get('type') == 'auth' and not authenticated:
                        # Handle authentication
                        host = data.get('host', SSH_HOST)
                        port = data.get('port', SSH_PORT)
                        username = data.get('username', DEFAULT_USER)
                        password = data.get('password', '')
                        
                        ssh = SSHSession()
                        try:
                            ssh.connect(host, port, username, password)
                            authenticated = True
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
        if ssh:
            ssh.close()
    
    logger.info("WebSocket connection closed")
    return ws


async def ssh_reader(ws, ssh):
    """Background task to read from SSH and send to WebSocket."""
    import json
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
    return web.json_response({'status': 'ok', 'ssh_host': SSH_HOST, 'ssh_port': SSH_PORT})


def create_app():
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/health', health_handler)
    return app


def main():
    """Start the SSH web terminal server."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           L.O.V.E. SSH Web Terminal Server                   ║
╠══════════════════════════════════════════════════════════════╣
║  Web UI:    http://localhost:{WEB_PORT:<5}                          ║
║  Default SSH: {DEFAULT_USER}@{SSH_HOST}:{SSH_PORT:<5}                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=WEB_PORT, print=None)


if __name__ == '__main__':
    main()
