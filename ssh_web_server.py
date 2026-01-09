#!/usr/bin/env python3
"""
SSH Web Terminal Server for L.O.V.E.
Serves a web page with xterm.js that connects via WebSocket to SSH.
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
SSH_USER = os.environ.get('SSH_USER', os.environ.get('USER', 'root'))
SSH_PASSWORD = os.environ.get('SSH_PASSWORD', '')  # Optional, will prompt if empty
SSH_KEY_PATH = os.environ.get('SSH_KEY_PATH', os.path.expanduser('~/.ssh/id_rsa'))

# HTML template with embedded xterm.js
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
            text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
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
        #terminal-container {
            flex: 1;
            padding: 15px;
            display: flex;
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
            <span id="status-text">Disconnected</span>
        </div>
    </header>
    <div id="terminal-container">
        <div id="terminal"></div>
    </div>
    <div class="info-bar">
        <span>Press Ctrl+Shift+V to paste | Ctrl+C to interrupt</span>
        <span id="connection-info">Connecting to SSH...</span>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/xterm-addon-web-links@0.9.0/lib/xterm-addon-web-links.min.js"></script>
    <script>
        const terminal = new Terminal({
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

        const fitAddon = new FitAddon.FitAddon();
        const webLinksAddon = new WebLinksAddon.WebLinksAddon();
        
        terminal.loadAddon(fitAddon);
        terminal.loadAddon(webLinksAddon);
        terminal.open(document.getElementById('terminal'));
        fitAddon.fit();

        window.addEventListener('resize', () => fitAddon.fit());

        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;

        function setStatus(status, text) {
            const dot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            dot.className = 'status-dot ' + status;
            statusText.textContent = text;
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            setStatus('connecting', 'Connecting...');
            terminal.writeln('\\r\\n\\x1b[33m[Connecting to SSH...]\\x1b[0m\\r\\n');

            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                setStatus('connected', 'Connected');
                document.getElementById('connection-info').textContent = 
                    `Connected to ${window.location.host}`;
                reconnectAttempts = 0;
                
                // Send terminal size
                const dims = { cols: terminal.cols, rows: terminal.rows };
                ws.send(JSON.stringify({ type: 'resize', ...dims }));
            };

            ws.onmessage = (event) => {
                terminal.write(event.data);
            };

            ws.onclose = (event) => {
                setStatus('', 'Disconnected');
                terminal.writeln('\\r\\n\\x1b[31m[Connection closed]\\x1b[0m');
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                    terminal.writeln(`\\x1b[33m[Reconnecting in ${delay/1000}s...]\\x1b[0m`);
                    setTimeout(connect, delay);
                } else {
                    terminal.writeln('\\x1b[31m[Max reconnection attempts reached. Refresh to try again.]\\x1b[0m');
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                terminal.writeln('\\r\\n\\x1b[31m[Connection error]\\x1b[0m');
            };
        }

        // Send terminal input to server
        terminal.onData((data) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'input', data: data }));
            }
        });

        // Handle terminal resize
        terminal.onResize(({ cols, rows }) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'resize', cols, rows }));
            }
        });

        // Initial connection
        connect();
        terminal.focus();
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
    
    def connect(self, host=SSH_HOST, port=SSH_PORT, username=SSH_USER, 
                password=SSH_PASSWORD, key_path=SSH_KEY_PATH):
        """Establish SSH connection."""
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Try key-based auth first, fall back to password
        try:
            if os.path.exists(key_path):
                logger.info(f"Attempting SSH key auth with {key_path}")
                self.client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    key_filename=key_path,
                    timeout=10
                )
            elif password:
                logger.info("Attempting SSH password auth")
                self.client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    password=password,
                    timeout=10
                )
            else:
                # Try with SSH agent or default keys
                logger.info("Attempting SSH auth with agent/default keys")
                self.client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    timeout=10
                )
        except Exception as e:
            logger.error(f"SSH connection failed: {e}")
            raise
        
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
    
    ssh = SSHSession()
    try:
        ssh.connect()
    except Exception as e:
        await ws.send_str(f"\r\n\x1b[31mSSH Connection Failed: {e}\x1b[0m\r\n")
        await ws.send_str("\x1b[33mCheck that SSH is running and credentials are correct.\x1b[0m\r\n")
        await ws.close()
        return ws
    
    # Background task to read from SSH and send to WebSocket
    async def ssh_reader():
        while not ws.closed and ssh.is_active():
            try:
                data = ssh.recv()
                if data:
                    await ws.send_str(data)
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"SSH read error: {e}")
                break
    
    reader_task = asyncio.create_task(ssh_reader())
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                import json
                try:
                    data = json.loads(msg.data)
                    if data.get('type') == 'input':
                        ssh.send(data.get('data', ''))
                    elif data.get('type') == 'resize':
                        cols = data.get('cols', 80)
                        rows = data.get('rows', 24)
                        ssh.resize(cols, rows)
                except json.JSONDecodeError:
                    # Raw text input
                    ssh.send(msg.data)
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break
    finally:
        reader_task.cancel()
        ssh.close()
    
    logger.info("WebSocket connection closed")
    return ws


async def index_handler(request):
    """Serve the main HTML page."""
    return web.Response(text=HTML_TEMPLATE, content_type='text/html')


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
║  SSH Host:  {SSH_HOST:<15} Port: {SSH_PORT:<5}                   ║
║  SSH User:  {SSH_USER:<15}                                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=WEB_PORT, print=None)


if __name__ == '__main__':
    main()
