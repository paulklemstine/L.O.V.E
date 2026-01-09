#!/bin/bash

# start_bridge.sh
# 1. Start SOCKS Proxy (Unix Socket) in background
# 2. Start SSH Daemon
# 3. Establish Reverse SSH Tunnel to Host

# Start SOCKS Proxy
if ! pgrep -f "vm_socks_proxy" > /dev/null; then
    echo "Starting VM SOCKS Proxy..."
    nohup python3 /usr/local/bin/vm_socks_proxy > /tmp/socks_proxy.log 2>&1 &
    # Wait for socket
    for i in {1..10}; do
        if [ -S /tmp/socks.sock ]; then break; fi
        sleep 0.5
    done
fi

# Ensure SSH Host Keys
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    echo "Generating SSH Host Keys..."
    ssh-keygen -A
fi

# Start SSH Server
mkdir -p /run/sshd
if ! pgrep -x "sshd" > /dev/null; then
    echo "Starting SSH Daemon..."
    /usr/sbin/sshd
fi

# Configuration
# HOST_IP should be reachable via the SOCKS proxy.
# Since the SOCKS proxy bridges to the HOST via websocket, 
# 'localhost' relative to the WS connection is the HOST.
# So "localhost" on the SOCKS proxy side (inside VM) -> maps to HOST's localhost.
# Wait, vm_socks_server maps requests to `host` variable.
# If we ask SOCKS to connect to "localhost", `ws_proxy.py` on FULL HOST will connect to "localhost" (which is HOST).
# So targeting "localhost" or "127.0.0.1" from INSIDE VM through SOCKS = HOST.

HOST_USER="raver1975" # User on the HOST machine
HOST_IP="127.0.0.1" # Connect to this IP *through* the proxy
HOST_SSH_PORT="22" # Port of SSH server on HOST

# Port to forward FROM Host TO VM
# Host:22222 -> VM:22
REMOTE_PORT="22222"

echo "Starting Reverse SSH Tunnel..."
echo "Connecting to ${HOST_USER}@${HOST_IP}:${HOST_SSH_PORT} via SOCKS..."
echo "Exposing VM:22 as Host:${REMOTE_PORT}"

# Use socks_connector.py as ProxyCommand
# We assume socks_connector.py is in /usr/local/bin/
PROXY_CMD="python3 /usr/local/bin/socks_connector.py %h %p"

# autossh would be better but simple loop for now
while true; do
    ssh -o ProxyCommand="$PROXY_CMD" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ExitOnForwardFailure=yes \
        -N -R ${REMOTE_PORT}:localhost:22 \
        ${HOST_USER}@${HOST_IP} -p ${HOST_SSH_PORT}
    
    echo "SSH Tunnel died, retrying in 5s..."
    sleep 5
done
