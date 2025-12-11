import sys
import os
import struct
import base64
import time
import argparse
from urllib.parse import urlparse

# Configuration
BRIDGE_IN_FILE = "/root/host_to_vm"
BRIDGE_OUT_PREFIX = "[[BRIDGE:"
BRIDGE_OUT_SUFFIX = "]]"

# Protocol Opcodes (Bridge V3)
OP_CONNECT = 1
OP_DATA = 2
OP_CLOSE = 3

def log(msg):
    sys.stderr.write(f"[pcurl] {msg}\n")
    sys.stderr.flush()

def send_packet(conn_id, opcode, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    header = struct.pack("<I B I", conn_id, opcode, len(payload))
    packet = header + payload
    b64 = base64.b64encode(packet).decode('utf-8')
    # Print to stdout so JS console sees it
    sys.stdout.write(f"{BRIDGE_OUT_PREFIX}{b64}{BRIDGE_OUT_SUFFIX}")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Python Bridge Curl")
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    target_url = args.url
    if not target_url.startswith(('http://', 'https://')):
        target_url = 'http://' + target_url
    
    parsed = urlparse(target_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    path = parsed.path or "/"
    
    conn_id = int(time.time()) % 10000 + 100 # Randomish ID
    
    if args.verbose:
        log(f"Connecting to {host}:{port}...")
        
    payload = struct.pack('>H', len(host)) + host.encode('utf-8') + struct.pack('>H', port)
    send_packet(conn_id, OP_CONNECT, payload)
    
    time.sleep(0.5) 
    
    req = f"GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: pcurl/1.0\r\nConnection: close\r\n\r\n"
    if args.verbose:
        log(f"Sending Request:\n{req.strip()}")
        
    send_packet(conn_id, OP_DATA, req.encode('utf-8'))
    
    # Ensure bridge file exists
    if not os.path.exists(BRIDGE_IN_FILE):
        open(BRIDGE_IN_FILE, 'w').close()
        
    f = open(BRIDGE_IN_FILE, 'rb')
    f.seek(0, 2)
    
    buffer = b""
    start_time = time.time()
    
    while True:
        chunk = f.read()
        if chunk:
            buffer += chunk
            while len(buffer) >= 9:
                cid, op, length = struct.unpack("<I B I", buffer[:9])
                if len(buffer) < 9 + length: break
                
                payload = buffer[9:9+length]
                buffer = buffer[9+length:]
                
                if cid == conn_id:
                    if op == OP_DATA:
                        sys.stdout.buffer.write(payload)
                        sys.stdout.flush()
                    elif op == OP_CLOSE:
                        if args.verbose: log("Connection closed.")
                        return
                    elif op == 0x81: # ERROR
                        log("Connection Error.")
                        return
        else:
            time.sleep(0.05)
            if time.time() - start_time > 30: # Timeout
                log("Timeout waiting for response.")
                return

if __name__ == "__main__":
    main()
