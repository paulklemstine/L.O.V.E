#!/usr/bin/env python3
# vm_net.py - The "Inside Man" SOCKS Server
import socket
import select
import struct
import sys
import threading
import os
import time

# Config
BRIDGE_FILE = "/mnt/bridge/netbridge" # Mounted DataDevice
SOCKS_HOST = "127.0.0.1"
SOCKS_PORT = 1080

# Protocol Opcodes
OP_CONNECT = 0x01
OP_DATA = 0x02
OP_CLOSE = 0x03

def log(msg):
    sys.stderr.write(f"[*] {msg}\n")
    sys.stderr.flush()

# Global map of connections
connections = {}
conn_lock = threading.Lock()
next_conn_id = 1

def reader_thread():
    """Reads from the bridge file and dispatches to sockets"""
    global connections
    
    log("Bridge reader starting...")
    
    while True:
        try:
            if not os.path.exists(BRIDGE_FILE):
                time.sleep(0.1)
                continue
                
            with open(BRIDGE_FILE, "rb") as f:
                data = f.read()
                
            if not data or len(data) == 0:
                time.sleep(0.01)
                continue
                
            # Clear the file after reading
            with open(BRIDGE_FILE, "wb") as f:
                f.write(b'')
                
            # Process the data
            buffer = data
            while len(buffer) >= 9:
                view_bytes = buffer[:9]
                conn_id, opcode, payload_len = struct.unpack('>IBI', view_bytes)
                
                if len(buffer) < 9 + payload_len:
                    break
                    
                payload = buffer[9:9+payload_len]
                buffer = buffer[9+payload_len:]
                
                with conn_lock:
                    if conn_id in connections:
                        sock = connections[conn_id]
                        if opcode == OP_DATA:
                            try:
                                sock.send(payload)
                            except:
                                pass
                        elif opcode == OP_CLOSE:
                            try:
                                sock.close()
                            except:
                                pass
                            del connections[conn_id]
                        elif opcode == 0x01: # CONNECTED confirmation
                            try:
                                sock.send(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
                            except:
                                pass
                        elif opcode == 0x81: # ERROR
                            try:
                                sock.send(b"\x05\x01\x00\x01\x00\x00\x00\x00\x00\x00")
                                sock.close()
                            except:
                                pass
                            if conn_id in connections:
                                del connections[conn_id]

        except Exception as e:
            log(f"Reader error: {e}")
            time.sleep(0.1)

def send_to_bridge(conn_id, opcode, data=b''):
    """Writes a framed packet to the bridge file"""
    # Format: [CONN_ID:4][OPCODE:1][LEN:4][DATA]
    try:
        packet = struct.pack('>IBI', conn_id, opcode, len(data)) + data
        with open(BRIDGE_FILE, "ab") as f:
            f.write(packet)
    except Exception as e:
        log(f"Write error: {e}")

def client_handler(client_socket, conn_id):
    try:
        # SOCKS handshake ...
        
        # 1. Greeting
        client_socket.recv(262) # Eat greeting (approx)
        client_socket.send(b"\x05\x00")
        
        # 2. Request
        header = client_socket.recv(4)
        if not header: return
        ver, cmd, rsv, atyp = header
        
        if cmd != 1: return # Only CONNECT
        
        dest_addr = ""
        dest_port = 0
        
        if atyp == 1:
            dest_addr = socket.inet_ntoa(client_socket.recv(4))
        elif atyp == 3:
            l = ord(client_socket.recv(1))
            dest_addr = client_socket.recv(l).decode()
        elif atyp == 4:
            return # No IPv6
            
        dest_port = struct.unpack('>H', client_socket.recv(2))[0]
        
        # 3. Send CONNECT to Bridge
        host_bytes = dest_addr.encode()
        # Payload: [HOST_LEN:2][HOST][PORT:2]
        payload = struct.pack(f'>H{len(host_bytes)}sH', len(host_bytes), host_bytes, dest_port)
        
        send_to_bridge(conn_id, OP_CONNECT, payload)
        
        # 4. Pump Data
        while True:
            data = client_socket.recv(4096)
            if not data: break
            send_to_bridge(conn_id, OP_DATA, data)
            
    except Exception as e:
        log(f"Client {conn_id} error: {e}")
    finally:
        send_to_bridge(conn_id, OP_CLOSE)
        with conn_lock:
            if conn_id in connections:
                del connections[conn_id]
        try:
            client_socket.close()
        except:
            pass

def main():
    global next_conn_id
    
    # Ensure bridge file exists (it should be a char device or pipe, but for testing we might need to create it?)
    # In CheerpX, it's a special device. We assume it exists.
    
    # Start Reader Thread
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SOCKS_HOST, SOCKS_PORT))
    server.listen(10)
    
    log(f"SOCKS5 Server listening on {SOCKS_HOST}:{SOCKS_PORT}")
    
    while True:
        client, addr = server.accept()
        conn_id = next_conn_id
        next_conn_id += 1
        
        with conn_lock:
            connections[conn_id] = client
            
        t = threading.Thread(target=client_handler, args=(client, conn_id), daemon=True)
        t.start()

if __name__ == "__main__":
    main()
