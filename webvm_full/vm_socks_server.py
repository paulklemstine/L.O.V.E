import sys
import socket
import select
import struct
import os
import time
import base64

# Configuration
BRIDGE_IN_FILE = "/root/host_to_vm"
BRIDGE_OUT_PREFIX = "[[BRIDGE:"
BRIDGE_OUT_SUFFIX = "]]"
SOCKS_PORT = 1080
BUFFER_SIZE = 8192

# Protocol Opcodes (Must match Bridge V3)
OP_CONNECT = 1
OP_DATA = 2
OP_CLOSE = 3

# Global state
connections = {}  # conn_id -> socket
next_conn_id = 1

def log(msg):
    sys.stderr.write(f"[VM-SOCKS] {msg}\n")
    sys.stderr.flush()

def encode_packet(conn_id, opcode, payload=b""):
    # Bridge V3 Packet: [ConnID:4][Opcode:1][Len:4][Payload:...]
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    
    header = struct.pack("<I B I", conn_id, opcode, len(payload))
    packet = header + payload
    b64 = base64.b64encode(packet).decode('utf-8')
    sys.stdout.write(f"{BRIDGE_OUT_PREFIX}{b64}{BRIDGE_OUT_SUFFIX}")
    sys.stdout.flush()

class SocksServer:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', SOCKS_PORT))
        self.server_socket.listen(50) # Increase backlog
        self.inputs = [self.server_socket]
        self.bridge_buffer = b""
        
        # Open bridge input file
        if not os.path.exists(BRIDGE_IN_FILE):
            try:
                open(BRIDGE_IN_FILE, 'w').close()
            except:
                pass
        
        self.bridge_in = None
        try:
            self.bridge_in = open(BRIDGE_IN_FILE, 'rb')
            self.bridge_in.seek(0, 2)
        except Exception as e:
            log(f"Warning: Could not open bridge file: {e}")

    def handle_handshake(self, sock):
        try:
            # SOCKS5 Greeting
            header = sock.recv(2)
            if not header or len(header) < 2: return False
            ver, nmethods = struct.unpack("!BB", header)
            if ver != 5: return False
            sock.recv(nmethods) # Consume methods
            sock.sendall(b"\x05\x00") # No Auth
            return True
        except:
            return False

    def handle_request(self, sock, conn_id):
        try:
            header = sock.recv(4)
            if len(header) < 4: return False
            ver, cmd, _, atyp = struct.unpack("!BBBB", header)
            
            if ver != 5 or cmd != 1: # CONNECT only
                # Command not supported
                sock.sendall(b"\x05\x07\x00\x01\x00\x00\x00\x00\x00\x00")
                return False
            
            addr = ""
            if atyp == 1: # IPv4
                addr = socket.inet_ntoa(sock.recv(4))
            elif atyp == 3: # Domain
                l = ord(sock.recv(1))
                addr = sock.recv(l).decode()
            else: # IPv6 (not supported by bridge nicely yet)
                sock.sendall(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")
                return False
                
            port = struct.unpack("!H", sock.recv(2))[0]
            log(f"Connect {conn_id}: {addr}:{port}")
            
            # Use Binary Connect Payload: [Len:2][Host][Port:2] (Big Endian)
            # This matches what pcurl.py sends and what the JS bridge expects
            host_bytes = addr.encode('utf-8')
            payload = struct.pack('>H', len(host_bytes)) + host_bytes + struct.pack('>H', port)
            
            encode_packet(conn_id, OP_CONNECT, payload)
            
            # Optimistic Generic Success to Client
            # The bridge is async, so we assume success for now. 
            # Real SOCKS would wait for upstream confirm.
            sock.sendall(b"\x05\x00\x00\x01" + socket.inet_aton("127.0.0.1") + struct.pack("!H", 1080))
            return True
        except Exception as e:
            log(f"Req Error: {e}")
            return False

    def check_bridge_input(self):
        if not self.bridge_in: return
        try:
            # Non-blocking read attempt logic (naive file read)
            # Since we are in a tight loop, we read what's there.
            data = self.bridge_in.read()
            if data:
                self.bridge_buffer += data
                self.process_buffer()
            else:
                # Clear EOF to allow reading more data appended later
                # Not strictly necessary for some fstreams but good for 'tail' behavior in Py
                pass
        except Exception as e:
            log(f"Bridge Read Error: {e}")

    def process_buffer(self):
        while len(self.bridge_buffer) >= 9:
            cid, op, length = struct.unpack("<I B I", self.bridge_buffer[:9])
            if len(self.bridge_buffer) < 9 + length: break
            
            payload = self.bridge_buffer[9:9+length]
            self.bridge_buffer = self.bridge_buffer[9+length:]
            
            if cid in connections:
                sock = connections[cid]
                try:
                    if op == OP_DATA:
                        sock.sendall(payload)
                    elif op == OP_CLOSE:
                        log(f"Bridge closed {cid}")
                        self.close_connection(cid)
                    elif op == 0x81: # ERROR
                        log(f"Bridge Error {cid}")
                        self.close_connection(cid)
                except:
                    self.close_connection(cid)

    def close_connection(self, cid):
        if cid in connections:
            sock = connections[cid]
            if sock in self.inputs:
                self.inputs.remove(sock)
            try:
                sock.close()
            except:
                pass
            del connections[cid]

    def run(self):
        global next_conn_id
        while True:
            self.check_bridge_input()
            
            # Short timeout to keep polling the file
            readable, _, _ = select.select(self.inputs, [], [], 0.01)
            
            for s in readable:
                if s is self.server_socket:
                    try:
                        c, _ = s.accept()
                        c.setblocking(1) # SOCKS handshake is sync
                        if self.handle_handshake(c):
                            cid = next_conn_id
                            next_conn_id += 1
                            if self.handle_request(c, cid):
                                c.setblocking(0) # Switch to async for data
                                connections[cid] = c
                                self.inputs.append(c)
                            else:
                                c.close()
                        else:
                            c.close()
                    except Exception as e:
                        log(f"Accept Error: {e}")
                else:
                    # Client data
                    current_cid = -1
                    for k, v in connections.items():
                        if v is s:
                            current_cid = k
                            break
                    
                    try:
                        data = s.recv(BUFFER_SIZE)
                        if data:
                            encode_packet(current_cid, OP_DATA, data)
                        else:
                            encode_packet(current_cid, OP_CLOSE)
                            self.close_connection(current_cid)
                    except:
                        encode_packet(current_cid, OP_CLOSE)
                        self.close_connection(current_cid)

if __name__ == "__main__":
    # Redirect stderr to /dev/null if verbose not wanted, 
    # but for now we keep it so we can see logs in console if needed.
    # Ideally logs go to specific file, but stderr is fine.
    log("Starting Socks5 Server...")
    SocksServer().run()
