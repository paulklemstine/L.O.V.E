# CheerpX Networking Breakout: A Developer's Guide

**By Antigravity Agent**

This document details the exact methodology used to "punch through" from a sandboxed CheerpX WebVM to the open Internet. It overcomes the current limitations of the CheerpX environment (specifically the lack of raw TCP socket support and `socket.accept()` failures) by implementing a custom full-duplex bridge over WebSocket.

---

## 1. The Challenge

CheerpX visualizes a Linux environment in the browser, but it runs isolated.
*   **No Direct TCP**: Standard `socket` syscalls do not map to browser networking API directly for arbitrary ports.
*   **No Listeners**: `bind()` and `accept()` are often stubbed or fail, making local proxy listeners (like SOCKS5) impossible to implement reliably inside the VM.
*   **Time Quirks**: The Python `time` module can suffer from `OverflowError` on `time.sleep()` due to time variable encoding in the emulated kernel.

## 2. The Solution: "Stdout Exfiltration & File Infiltration"

We created a custom protocol (**Bridge V3**) that tunnels TCP data through the available I/O channels:
1.  **Outbound (VM -> Host)**: Data is Base64-encoded, wrapped in a specific tag (`[[BRIDGE:...]]`), and printed to `stdout` / `stderr`. A JavaScript `MutationObserver` intercepts these tags from the Console DOM.
2.  **Inbound (Host -> VM)**: Data returned from the host is written by the JavaScript layer into a file inside the VM (`/root/host_to_vm`) using `cx.run()`. The VM client polls this file.
3.  **Transport**: The JavaScript layer forwards these packets to a Python WebSocket proxy running on the Host OS, which performs the actual networking.

### Architecture Diagram

```ascii
[ VM Process (pcurl.py) ]
       |
       | (Stdout: [[BRIDGE:B64_DATA]])
       v
[ Browser DOM (#console) ] <--- MutationObserver
       |
       | (JS parses tag)
       v
[ index.html (Bridge Logic) ]
       |
       | (WebSocket ws://localhost:8082)
       v
[ Host OS (ws_proxy.py) ] -----> [ Internet (google.com:80) ]
```

---

## 3. The Implementation (Step-by-Step)

To recreate this breakout, you need three components.

### Component A: The Host Proxy (`ws_proxy.py`)

This script runs on the machine hosting the WebVM. It translates WebSocket messages into TCP connections.

**Code (`ws_proxy.py`):**
```python
import asyncio
import websockets
import socket
import struct
import logging

async def handle_connection(websocket):
    connection_id = id(websocket)
    try:
        async for message in websocket:
            if len(message) < 1: continue
            cmd = message[0]
            data = message[1:]

            if cmd == 0x01: # CONNECT
                # Parse [host_len:2][host][port:2]
                host_len = struct.unpack('>H', data[0:2])[0]
                host = data[2:2+host_len].decode('utf-8')
                port = struct.unpack('>H', data[2+host_len:4+host_len])[0]
                
                # Perform actual TCP Connection
                reader, writer = await asyncio.open_connection(host, port)
                
                # Send Success to JS
                await websocket.send(bytes([0x01])) 
                
                # Pipe TCP -> WS
                asyncio.create_task(tcp_to_ws(reader, websocket))
                
            elif cmd == 0x02: # DATA
                # Write to TCP (writer saved in closure/map)
                writer.write(data)
                await writer.drain()

async def tcp_to_ws(reader, websocket):
    while True:
        data = await reader.read(4096)
        if not data: break
        # Send [CMD_DATA][PAYLOAD]
        await websocket.send(bytes([0x02]) + data)

async def main():
    async with websockets.serve(handle_connection, "0.0.0.0", 8082):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

### Component B: The Browser Bridge (`index.html`)

The HTML file acts as the router. It observes the console and writes responses back to the VM.

**Key Logic:**

1.  **Console Observer**:
    ```javascript
    const observer = new MutationObserver((mutations) => {
        // Concatenate text content
        // Regex match /\[\[BRIDGE:(.*?)\]\]/
        // Decode Base64 -> Send to WebSocket
    });
    ```

2.  **Writing to VM**:
    When the WebSocket receives data, we write it to the VM's filesystem using `cx.run`.
    ```javascript
    function sendToVm(data) {
        // data is Uint8Array
        // Convert to B64
        const b64 = ...; 
        // Append to file
        cx.run("/bin/sh", ["-c", `echo "${b64}" | base64 -d >> /root/host_to_vm`], { uid: 0, gid: 0 });
    }
    ```

### Component C: The VM Client (`pcurl.py`)

The Python script inside the VM. It mimics `curl` but talks "Bridge V3".

**Critical Workaround: Busy Wait**
CheerpX's `time.sleep` crashes with `OverflowError`. We use this instead:
```python
def busy_wait(loops=10000):
    for _ in range(loops): pass
```

**Client Code Structure:**
```python
# 1. Construct Packet
payload = struct.pack('>H', len(host)) + host.encode() + struct.pack('>H', port)
header = struct.pack("<I B I", conn_id, 1, len(payload)) # ID, OP_CONNECT, LEN

# 2. Exfiltrate
b64 = base64.b64encode(header + payload).decode()
sys.stdout.write(f"[[BRIDGE:{b64}]]")
sys.stdout.flush()

# 3. Poll for Response
f = open("/root/host_to_vm", "rb")
while True:
    chunk = f.read()
    if not chunk:
        busy_wait(100)
        continue
    # Parse chunk...
```

---

## 4. How to Recreate

1.  **Dependencies**:
    *   Host: `python3`, `pip install websockets`
    *   VM: CheerpX image (provided).

2.  **Setup Host**:
    Run `python3 ws_proxy.py`. Verify it listens on 8082.

3.  **Setup Browser**:
    Host your `index.html` on a web server (e.g., `python3 -m http.server 8080`).
    Open `http://localhost:8080`.

4.  **Inject Logic**:
    Your `index.html` must contain the `injectDirectClient()` function which writes the `pcurl.py` code into `/root/pcurl.py` immediately after boot.

5.  **Run**:
    Wait for boot.
    The client is automatically injected.
    Run `curl google.com`.

## 5. Technical Insights

*   **Latency**: The file polling adds some latency, but is surprisingly fast for text/HTTP.
*   **Throughput**: Limited by `MutationObserver` speed and `cx.run` process spawning overhead. Optimized by batching writes.
*   **Reliability**: Using standard `curl` aliases ensures users use the improved logic without knowing standard tools are broken.
