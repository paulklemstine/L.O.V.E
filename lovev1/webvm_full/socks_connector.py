
import socket
import sys
import struct
import select
import os

# usage: python3 socks_connector.py <host> <port>
# connects to unix socket at /tmp/socks.sock and performs SOCKS5 connect

SOCKS_PATH = "/tmp/socks.sock"

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: socks_connector.py <host> <port>\n")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    # Connect to SOCKS server via Unix Socket
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCKS_PATH)
    except Exception as e:
        sys.stderr.write(f"Failed to connect to unix socket {SOCKS_PATH}: {e}\n")
        sys.exit(1)

    # 1. Handshake
    # Client sends: VER(5) NMETHODS(1) METHODS(0 - No Auth)
    s.sendall(b"\x05\x01\x00")
    
    # Server replies: VER(5) METHOD(0)
    resp = s.recv(2)
    if len(resp) < 2 or resp[0] != 0x05 or resp[1] != 0x00:
        sys.stderr.write("SOCKS5 handshake failed or auth required\n")
        s.close()
        sys.exit(1)

    # 2. Connect Request
    # Client sends: VER(5) CMD(1-Connect) RSV(0) ATYP DST.ADDR DST.PORT
    # We use ATYP=3 (Domain name) for simplicity to let proxy resolve, or just send IP as string if it's IP.
    # Logic: if host is IP, use simple parsing, but domain works for everything usually in SOCKS5 if proxy supports it.
    # vm_socks_server.py supports ATYP 1 (IPv4) and 3 (Domain).
    # Let's use ATYP 3 always for flexibility unless it's strictly IPv4 bytes.
    # Actually better to use ATYP 3 for strings.
    
    host_bytes = host.encode('utf-8')
    req = struct.pack("!BBBB", 5, 1, 0, 3) # VER, CMD, RSV, ATYP=3
    req += struct.pack("!B", len(host_bytes))
    req += host_bytes
    req += struct.pack("!H", port)
    
    s.sendall(req)

    # 3. Response
    # Server sends: VER(5) REP(0-Success) RSV(0) ATYP BND.ADDR BND.PORT
    resp = s.recv(4)
    if len(resp) < 4:
        sys.stderr.write("SOCKS5 connect response too short\n")
        s.close()
        sys.exit(1)
        
    ver, rep, rsv, atyp = struct.unpack("!BBBB", resp)
    if rep != 0:
        sys.stderr.write(f"SOCKS5 connect failed with error: {rep}\n")
        s.close()
        sys.exit(1)
        
    # Skip remaining BND address bytes
    if atyp == 1:
        s.recv(4 + 2)
    elif atyp == 3:
        l = ord(s.recv(1))
        s.recv(l + 2)
    elif atyp == 4:
        s.recv(16 + 2)
        
    # 4. Proxy Loop
    # Bridge stdin/stdout to socket
    # Using select for non-blocking I/O
    
    inputs = [s, sys.stdin]
    
    try:
        while True:
            readable, _, _ = select.select(inputs, [], [])
            
            for r in readable:
                if r is s:
                    data = s.recv(4096)
                    if not data:
                        return # Connection closed
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                elif r is sys.stdin:
                    data = sys.stdin.buffer.read(4096)
                    if not data:
                        # EOF from stdin, shutdown send side of socket
                        s.shutdown(socket.SHUT_WR)
                        inputs.remove(sys.stdin)
                    else:
                        s.sendall(data)
                        
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
    finally:
        s.close()

if __name__ == "__main__":
    main()
