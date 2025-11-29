# WebSocket SOCKS5 Proxy for WebVM

This proxy enables network access for the CheerpX WebVM by providing a WebSocket-based SOCKS5 proxy.

## Requirements

```bash
pip install websockets
```

## Usage

1. Start the proxy server:
```bash
python3 ws_socks5_proxy.py
```

2. The proxy will listen on `ws://localhost:8001`

3. Configure your WebVM to use this proxy (see index.html)

## How it Works

- WebVM connects to the proxy via WebSocket
- Proxy implements SOCKS5 protocol over WebSocket
- TCP connections are tunneled through the WebSocket
- Enables git, curl, wget, and other network tools in the WebVM

## Security Note

This proxy allows the WebVM to make arbitrary network connections. Only run this on trusted networks or add authentication/filtering as needed.
