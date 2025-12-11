import socket
import sys

print("Starting Echo Server on 9999...")
sys.stdout.flush()

try:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 9999))
    s.listen(1)
    print("LISTENING")
    sys.stdout.flush()
    
    conn, addr = s.accept()
    print(f"ACCEPTED {addr}")
    sys.stdout.flush()
    
    while True:
        data = conn.recv(1024)
        if not data: break
        print(f"RECEIVED: {data}")
        conn.send(data)
        sys.stdout.flush()
        
    conn.close()
    s.close()
    print("CLOSED")
except Exception as e:
    print(f"ERROR: {e}")
    sys.stdout.flush()
