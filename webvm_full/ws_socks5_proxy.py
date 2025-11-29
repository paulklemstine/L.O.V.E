#!/usr/bin/env python3
"""
WebSocket SOCKS5 Proxy for CheerpX
Provides network access to WebVM through a WebSocket-based SOCKS5 proxy
"""

import asyncio
import websockets
import socket
import struct
import logging
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SOCKS5Proxy:
    def __init__(self):
        self.connections: Dict[int, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        logger.info(f"New client connected: {client_id}")
        
        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    continue
                
                await self.handle_socks5_message(websocket, client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            await self.cleanup_connection(client_id)
    
    async def handle_socks5_message(self, websocket, client_id, data):
        """Handle SOCKS5 protocol messages"""
        if len(data) < 1:
            return
        
        version = data[0]
        
        if version == 5:  # SOCKS5
            if len(data) >= 2:
                cmd = data[1]
                
                if cmd == 0:  # Auth methods
                    # No authentication required
                    await websocket.send(bytes([5, 0]))
                    
                elif cmd == 1:  # CONNECT
                    await self.handle_connect(websocket, client_id, data)
                    
        elif client_id in self.connections:
            # Forward data to target
            _, writer = self.connections[client_id]
            writer.write(data)
            await writer.drain()
    
    async def handle_connect(self, websocket, client_id, data):
        """Handle SOCKS5 CONNECT command"""
        try:
            # Parse SOCKS5 CONNECT request
            # Format: VER CMD RSV ATYP DST.ADDR DST.PORT
            if len(data) < 10:
                await websocket.send(bytes([5, 1, 0, 1, 0, 0, 0, 0, 0, 0]))  # General failure
                return
            
            atyp = data[3]
            
            if atyp == 1:  # IPv4
                host = socket.inet_ntoa(data[4:8])
                port = struct.unpack('>H', data[8:10])[0]
            elif atyp == 3:  # Domain name
                domain_len = data[4]
                host = data[5:5+domain_len].decode('utf-8')
                port = struct.unpack('>H', data[5+domain_len:7+domain_len])[0]
            else:
                await websocket.send(bytes([5, 8, 0, 1, 0, 0, 0, 0, 0, 0]))  # Address type not supported
                return
            
            logger.info(f"CONNECT request to {host}:{port}")
            
            # Connect to target
            try:
                reader, writer = await asyncio.open_connection(host, port)
                self.connections[client_id] = (reader, writer)
                
                # Send success response
                await websocket.send(bytes([5, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
                
                # Start forwarding data from target to client
                asyncio.create_task(self.forward_to_client(websocket, client_id, reader))
                
            except Exception as e:
                logger.error(f"Failed to connect to {host}:{port}: {e}")
                await websocket.send(bytes([5, 5, 0, 1, 0, 0, 0, 0, 0, 0]))  # Connection refused
                
        except Exception as e:
            logger.error(f"Error in handle_connect: {e}", exc_info=True)
            await websocket.send(bytes([5, 1, 0, 1, 0, 0, 0, 0, 0, 0]))  # General failure
    
    async def forward_to_client(self, websocket, client_id, reader):
        """Forward data from target to WebSocket client"""
        try:
            while True:
                data = await reader.read(8192)
                if not data:
                    break
                await websocket.send(data)
        except Exception as e:
            logger.error(f"Error forwarding to client {client_id}: {e}")
        finally:
            await self.cleanup_connection(client_id)
    
    async def cleanup_connection(self, client_id):
        """Clean up connection resources"""
        if client_id in self.connections:
            _, writer = self.connections[client_id]
            writer.close()
            try:
                await writer.wait_closed()
            except:
                pass
            del self.connections[client_id]
            logger.info(f"Cleaned up connection {client_id}")

async def main():
    """Start the WebSocket SOCKS5 proxy server"""
    proxy = SOCKS5Proxy()
    
    logger.info("Starting WebSocket SOCKS5 proxy on ws://0.0.0.0:8001")
    logger.info("This proxy enables network access for CheerpX WebVM")
    
    async with websockets.serve(
        proxy.handle_client,
        "0.0.0.0",
        8001,
        ping_interval=20,
        ping_timeout=10
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down proxy server")
