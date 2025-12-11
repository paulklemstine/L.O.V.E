# WebVM to Internet Networking Guide

This guide details the exact steps and code required to enable outbound network connectivity (e.g., `curl`) from a CheerpX WebVM to the internet, bypassing the limitations of browser-based networking and CheerpX's `socket.accept()` issues.

## Architecture Overview

Network requests flow as follows:
`VM (pcurl.py)` -> `(Exfil: Stdout [[BRIDGE...]])` -> `Browser (index.html)` -> 
`(WebSocket: 8082)` -> `Host (ws_proxy.py)` -> `Internet`

Return path:
`Internet` -> `Host (ws_proxy.py)` -> `WebSocket` -> `Browser` -> `(File: /root/host_to_vm)` -> `VM (pcurl.py)`

## Components

### 1. Host Proxy (`ws_proxy.py`)
Runs on the host machine. It accepts WebSocket connections from the browser, parses binary packets, and creates actual TCP connections to the target.

**Key Features:**
*   Listens on `0.0.0.0:8082`.
*   Protocol: `[CMD:1][DATA...]`.
*   CMD 1 (CONNECT): `[LEN_HOST:2][HOST][PORT:2]`.
*   CMD 2 (DATA): `[DATA]`.
*   CMD 3 (CLOSE).

### 2. Browser Bridge (`index.html`)
Injects the client script and acts as the middleman.

**Key Functions:**
*   `injectDirectClient()`: Writes `pcurl.py` to `/root/pcurl.py` via `cx.run`.
*   `setupConsoleBridge()`: Watches `console` div for `[[BRIDGE:<base64>]]` messages from VM.
*   `sendToVm(data)`: Writes response data to `/root/host_to_vm` for the VM to read.
*   `auto-inject`: Called before starting `bash` to ensure tools are ready immediately.

### 3. VM Client (`pcurl.py`)
A Python script running inside CheerpX that behaves like `curl` (basic HTTP GET support).

**Key Logic:**
*   Parses URL (defaults to `http://` if missing).
*   Constructs Bridge V3 packets.
*   Encodes them as Base64 wrapped in `[[BRIDGE:...]]` and prints to stdout.
*   Polls `/root/host_to_vm` for binary response data.
*   Uses busy-wait loops to avoid `time.sleep` overflow errors in CheerpX.

## Step-by-Step Implementation

### Step 1: Host Proxy
Ensure `ws_proxy.py` is running on the host:
```bash
python3 webvm_full/ws_proxy.py
```

### Step 2: WebVM Configuration
Ensure `index.html` has the correct `BRIDGE_WS_URL`:
```javascript
const BRIDGE_WS_URL = "ws://localhost:8082";
```
And `PYTHON_CLIENT_SRC` contains the robust `pcurl.py` code.

### Step 3: Usage
1.  Load `index.html` in browser.
2.  Wait for "CheerpX ready!".
3.  The system automatically injects tools.
4.  In the VM console:
    ```bash
    curl -v google.com
    ```

## Troubleshooting
*   **No Response**: Check `ws_proxy.log` on host. If empty, WebSocket connection failed (check port blocking).
*   **OverflowError**: Ensure `pcurl.py` uses iteration loops, not `time.sleep`.
*   **Permission Denied**: Ensure `/root/host_to_vm` is chmod 666 (handled by injection script).
*   **TypeError (NoneType)**: Ensure `pcurl.py` has the URL scheme patching logic.
