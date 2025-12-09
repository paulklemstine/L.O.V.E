#!/usr/bin/env python3
import time
import os
import struct
import sys

# Protocol Config
# STREAM PROTOCOL (Append Only)
# /mnt/bridge/vm_to_host: VM writes (Appends), Host reads (Offsets)
# /mnt/bridge/host_to_vm: Host writes (Appends), VM reads (Offsets)

BASE_DIR = "/mnt/bridge"
PIPE_OUT = os.path.join(BASE_DIR, "vm_to_host")
PIPE_IN  = os.path.join(BASE_DIR, "host_to_vm")

# Opcodes
OP_CONNECT = 0x01
OP_DATA    = 0x02
OP_CLOSE   = 0x03

def log(msg):
    sys.stderr.write(f"[DirectClient] {msg}\n")
    sys.stderr.flush()

def ensure_files():
    # Create output file if missing (though we open 'ab', so auto created)
    # But input file must exist to read
    if not os.path.exists(PIPE_IN):
        # log(f"Waiting for {PIPE_IN}...")
        # Since we are client, maybe we create it empty?
        # Host might rely on it existing? NO, Host creates it.
        # But for 'tail' logic, we handle missing file by waiting.
        pass

def send_frame(opcode, payload=b""):
    # Format: [OPCODE:1][LEN:4][PAYLOAD]
    conn_id = 999 
    packet = struct.pack('>IBI', conn_id, opcode, len(payload)) + payload
    
    try:
        with open(PIPE_OUT, "ab") as f:
            f.write(packet)
            f.flush()
    except Exception as e:
        log(f"Write error: {e}")

def read_response(offset, timeout=5):
    start = time.time()
    
    while time.time() - start < timeout:
        if os.path.exists(PIPE_IN):
            try:
                with open(PIPE_IN, "rb") as f:
                    f.seek(0, 2) # End
                    file_size = f.tell()
                    
                    if file_size > offset:
                        f.seek(offset)
                        data = f.read()
                        new_offset = offset + len(data)
                        return data, new_offset
                        
            except Exception as e:
                # log(f"Read error: {e}")
                pass
                
        time.sleep(0.1)
        
    return None, offset # Timeout

def main():
    target_host = "google.com"
    target_port = 80
    
    log(f"Connecting to {target_host} via Stream Bridge...")
    ensure_files()
    
    # 1. CONNECT
    host_bytes = target_host.encode()
    payload = struct.pack(f'>H{len(host_bytes)}sH', len(host_bytes), host_bytes, target_port)
    send_frame(OP_CONNECT, payload)
    
    log("Sent CONNECT. Sending GET...")
    time.sleep(1.0) 
    
    # 2. GET
    request = f"GET / HTTP/1.1\r\nHost: {target_host}\r\nConnection: close\r\n\r\n".encode()
    send_frame(OP_DATA, request)
    
    # 3. READ
    log("Waiting for response...")
    total_data = b""
    read_buffer = b""
    offset = 0
    
    start_wait = time.time()
    while time.time() - start_wait < 10: # Overall timeout
        chunk, new_offset = read_response(offset, timeout=1)
        offset = new_offset
        
        if chunk:
            read_buffer += chunk
            # Process Buffer
            while len(read_buffer) >= 9:
                conn_id, opcode, payload_len = struct.unpack('>IBI', read_buffer[:9])
                
                if len(read_buffer) >= 9 + payload_len:
                    payload = read_buffer[9:9+payload_len]
                    read_buffer = read_buffer[9+payload_len:] # Consume
                    
                    if conn_id != 999: continue
                    
                    if opcode == OP_DATA:
                        total_data += payload
                    elif opcode == OP_CLOSE:
                        log("Connection closed by host")
                        start_wait = 0 # Exit loop
                        break
                else:
                    break # Wait for more data
        else:
             # No new data in timeout window. Continue waiting or break if timed out?
             # read_response waits 1s.
             pass
             
    if total_data:
        log("SUCCESS! Response:")
        print("-" * 40)
        try:
            print(total_data.decode('utf-8', errors='ignore')[:500] + "...")
        except:
             print(total_data[:500])
        print("-" * 40)
    else:
        log("FAILED: No data.")

if __name__ == "__main__":
    main()
