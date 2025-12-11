#!/usr/bin/env python3
"""
WebSocket to TCP Proxy for CheerpX networking
Allows the WebVM to make network connections through a WebSocket tunnel
"""

import asyncio
import websockets
import socket
import struct
import logging
import sys

# Force stdout/stderr flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Maximum connections
MAX_CONNECTIONS = 100
active_connections = {}

def log_debug(msg):
    print(f"[DEBUG] {msg}", flush=True)

async def handle_connection(websocket):
    """Handle a WebSocket connection and proxy it to TCP"""
    connection_id = id(websocket)
    log_debug(f"New WebSocket connection: {connection_id}")
    
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Parse the message format: [command:1byte][data]
                if len(message) < 1:
                    continue
                    
                command = message[0]
                data = message[1:]
                
                log_debug(f"RX Command: {command}, Len: {len(data)}")

                if command == 0x01:  # CONNECT
                    # Format: [host_len:2bytes][host][port:2bytes]
                    if len(data) < 4:
                        log_debug("Conn packet too short")
                        await websocket.send(bytes([0x81]))  # ERROR
                        continue
                    
                    host_len = struct.unpack('>H', data[0:2])[0]
                    log_debug(f"Host Len: {host_len}")
                    
                    if len(data) < 2 + host_len + 2:
                        log_debug("Conn packet incomplete")
                        await websocket.send(bytes([0x81]))  # ERROR
                        continue
                    
                    host = data[2:2+host_len].decode('utf-8')
                    port = struct.unpack('>H', data[2+host_len:4+host_len])[0]
                    
                    log_debug(f"CONNECT request to {host}:{port}")
                    
                    try:
                        # Create TCP connection
                        start_time = asyncio.get_event_loop().time()
                        reader, writer = await asyncio.wait_for(
                            asyncio.open_connection(host, port), 
                            timeout=10.0
                        )
                        dt = asyncio.get_event_loop().time() - start_time
                        log_debug(f"Connected to {host}:{port} in {dt:.2f}s")
                        
                        active_connections[connection_id] = (reader, writer)
                        
                        # Send success
                        await websocket.send(bytes([0x01]))  # CONNECTED
                        log_debug(f"Sent CONNECTED ack")
                        
                        # Start reading from TCP and sending to WebSocket
                        asyncio.create_task(tcp_to_ws(reader, websocket, connection_id))
                        
                    except Exception as e:
                        log_debug(f"Failed to connect to {host}:{port}: {e}")
                        await websocket.send(bytes([0x81]))  # ERROR
                
                elif command == 0x02:  # SEND DATA
                    log_debug(f"Sending {len(data)} bytes to target")
                    if connection_id in active_connections:
                        _, writer = active_connections[connection_id]
                        writer.write(data)
                        await writer.drain()
                    else:
                        log_debug("No active connection for ID")
                
                elif command == 0x03:  # CLOSE
                    log_debug("Closing connection")
                    if connection_id in active_connections:
                        _, writer = active_connections[connection_id]
                        writer.close()
                        await writer.wait_closed()
                        del active_connections[connection_id]
                        await websocket.send(bytes([0x03]))  # CLOSED
                        
    except websockets.exceptions.ConnectionClosed:
        log_debug(f"WebSocket connection closed: {connection_id}")
    except Exception as e:
        log_debug(f"Error handling connection: {e}")
    finally:
        if connection_id in active_connections:
            _, writer = active_connections[connection_id]
            writer.close()
            del active_connections[connection_id]

async def tcp_to_ws(reader, websocket, connection_id):
    """Read from TCP and send to WebSocket"""
    log_debug("Started TCP reader task")
    try:
        while True:
            data = await reader.read(8192)
            if not data:
                log_debug("TCP connection closed by remote (EOF)")
                break
            
            log_debug(f"Read {len(data)} bytes from TCP, relaying to WS")
            # Send data with DATA command
            await websocket.send(bytes([0x02]) + data)
    except Exception as e:
        log_debug(f"Error in tcp_to_ws: {e}")
    finally:
        try:
            await websocket.send(bytes([0x03]))  # CLOSED
            log_debug("Sent CLOSED to WS")
        except:
            pass

async def main():
    """Start the WebSocket proxy server"""
    log_debug("Starting WebSocket proxy on ws://0.0.0.0:8082")
    async with websockets.serve(handle_connection, "0.0.0.0", 8082):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
