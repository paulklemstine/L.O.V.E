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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum connections
MAX_CONNECTIONS = 100
active_connections = {}

async def handle_connection(websocket, path):
    """Handle a WebSocket connection and proxy it to TCP"""
    connection_id = id(websocket)
    logger.info(f"New WebSocket connection: {connection_id}")
    
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Parse the message format: [command:1byte][data]
                if len(message) < 1:
                    continue
                    
                command = message[0]
                data = message[1:]
                
                if command == 0x01:  # CONNECT
                    # Format: [host_len:2bytes][host][port:2bytes]
                    if len(data) < 4:
                        await websocket.send(bytes([0x81]))  # ERROR
                        continue
                    
                    host_len = struct.unpack('>H', data[0:2])[0]
                    if len(data) < 2 + host_len + 2:
                        await websocket.send(bytes([0x81]))  # ERROR
                        continue
                    
                    host = data[2:2+host_len].decode('utf-8')
                    port = struct.unpack('>H', data[2+host_len:4+host_len])[0]
                    
                    logger.info(f"CONNECT to {host}:{port}")
                    
                    try:
                        # Create TCP connection
                        reader, writer = await asyncio.open_connection(host, port)
                        active_connections[connection_id] = (reader, writer)
                        
                        # Send success
                        await websocket.send(bytes([0x01]))  # CONNECTED
                        
                        # Start reading from TCP and sending to WebSocket
                        asyncio.create_task(tcp_to_ws(reader, websocket, connection_id))
                        
                    except Exception as e:
                        logger.error(f"Failed to connect to {host}:{port}: {e}")
                        await websocket.send(bytes([0x81]))  # ERROR
                
                elif command == 0x02:  # SEND DATA
                    if connection_id in active_connections:
                        _, writer = active_connections[connection_id]
                        writer.write(data)
                        await writer.drain()
                
                elif command == 0x03:  # CLOSE
                    if connection_id in active_connections:
                        _, writer = active_connections[connection_id]
                        writer.close()
                        await writer.wait_closed()
                        del active_connections[connection_id]
                        await websocket.send(bytes([0x03]))  # CLOSED
                        
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {connection_id}")
    except Exception as e:
        logger.error(f"Error handling connection: {e}")
    finally:
        # Clean up any active TCP connections
        if connection_id in active_connections:
            _, writer = active_connections[connection_id]
            writer.close()
            try:
                await writer.wait_closed()
            except:
                pass
            del active_connections[connection_id]

async def tcp_to_ws(reader, websocket, connection_id):
    """Read from TCP and send to WebSocket"""
    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break
            
            # Send data with DATA command
            await websocket.send(bytes([0x02]) + data)
    except Exception as e:
        logger.error(f"Error in tcp_to_ws: {e}")
    finally:
        # Send close notification
        try:
            await websocket.send(bytes([0x03]))  # CLOSED
        except:
            pass

async def main():
    """Start the WebSocket proxy server"""
    logger.info("Starting WebSocket proxy on ws://localhost:8001")
    async with websockets.serve(handle_connection, "0.0.0.0", 8001):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
