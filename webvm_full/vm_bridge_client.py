import sys
import os
import struct
import base64
import time

# Configuration
BRIDGE_IN_FILE = "/root/host_to_vm"
BRIDGE_OUT_PREFIX = "[[BRIDGE:"
BRIDGE_OUT_SUFFIX = "]]"

# Protocol Opcodes (Bridge V3)
OP_CONNECT = 1
OP_DATA = 2
OP_CLOSE = 3

def log(msg):
    sys.stderr.write(f"[VM-CLIENT] {msg}\n")
    sys.stderr.flush()

def send_packet(conn_id, opcode, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    header = struct.pack("<I B I", conn_id, opcode, len(payload))
    packet = header + payload
    b64 = base64.b64encode(packet).decode('utf-8')
    print(f"{BRIDGE_OUT_PREFIX}{b64}{BRIDGE_OUT_SUFFIX}", flush=True)

def test_google():
    conn_id = 100
    host = "google.com"
    port = 80
    
    log(f"Connecting to {host}:{port} (ID {conn_id})...")
    
    # 1. Connect Payload: [len:2][host][port:2] (Big Endian)
    payload = struct.pack('>H', len(host)) + host.encode('utf-8') + struct.pack('>H', port)
    send_packet(conn_id, OP_CONNECT, payload)
    
    # 2. Wait for connection (Blind wait)
    time.sleep(1)
    
    # 3. Send Request
    req = b"GET / HTTP/1.0\r\nHost: google.com\r\n\r\n"
    log("Sending GET request...")
    send_packet(conn_id, OP_DATA, req)
    
    # 4. Read Loop
    log("Reading response...")
    
    if not os.path.exists(BRIDGE_IN_FILE):
        open(BRIDGE_IN_FILE, 'w').close()
        
    f = open(BRIDGE_IN_FILE, 'rb')
    f.seek(0, 2)
    
    buffer = b""
    start_time = time.time()
    
    while time.time() - start_time < 10:
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
                        try:
                            # Try decoding as text, if fail show binary repr
                            text = payload.decode()
                            print(text, end='')
                        except:
                            print(f"<BINARY {len(payload)} bytes>", end='')
                            
                    elif op == OP_CLOSE:
                        log("\nConnection closed by remote.")
                        return
                    elif op == 0x81: # ERROR
                        log("\nConnection Error from Bridge.")
                        return
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    test_google()
