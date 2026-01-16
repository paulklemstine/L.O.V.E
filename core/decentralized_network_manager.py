import os
import json
import asyncio
from aiohttp import web
import aiohttp

class DecentralizedNetworkManager:
    """
    Manages peer-to-peer communication for the decentralized network.
    This manager handles peer discovery, message broadcasting, and direct messaging.
    """
    def __init__(self, host='0.0.0.0', port=8989):
        self.host = host
        self.port = port
        self.peers = set()
        self.server = None
        self.received_messages = []

    async def _handle_handshake(self, request):
        """Handles incoming handshake requests from new peers."""
        data = await request.json()
        peer_address = data.get("address")
        if peer_address:
            self.peers.add(peer_address)
            print(f"Handshake successful with {peer_address}")
            return web.json_response({"status": "ok", "peers": list(self.peers)})
        return web.json_response({"status": "error", "message": "Invalid handshake"}, status=400)


    async def _handle_broadcast(self, request):
        """Handles incoming broadcast messages."""
        data = await request.json()
        self.received_messages.append(data)
        print(f"Received broadcast: {data}")
        return web.json_response({"status": "broadcast received"})

    async def start_server(self):
        """Starts the P2P server to listen for incoming connections."""
        app = web.Application()
        app.router.add_post('/handshake', self._handle_handshake)
        app.router.add_post('/broadcast', self._handle_broadcast)

        runner = web.AppRunner(app)
        await runner.setup()
        self.server = web.TCPSite(runner, self.host, self.port)
        await self.server.start()
        print(f"P2P server started on {self.host}:{self.port}")

    async def stop_server(self):
        """Stops the P2P server."""
        if self.server:
            await self.server.stop()
            print("P2P server stopped.")

    async def connect_to_peer(self, peer_host, peer_port):
        """Connects to a new peer and performs a handshake."""
        peer_address = f"http://{peer_host}:{peer_port}"
        my_address = f"http://{self.host}:{self.port}"
        payload = {"address": my_address}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{peer_address}/handshake", json=payload) as response:
                    if response.status == 200:
                        self.peers.add(peer_address)
                        print(f"Successfully connected to peer: {peer_address}")
                        return True
                    else:
                        print(f"Failed to connect to peer {peer_address}: Status {response.status}")
                        return False
        except aiohttp.ClientConnectorError as e:
            print(f"Failed to connect to peer {peer_address}: {e}")
            return False

    async def broadcast_message(self, message):
        """Broadcasts a message to all known peers."""
        if not isinstance(message, dict):
            message = {"content": message}

        for peer in self.peers:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{peer}/broadcast", json=message) as response:
                        if response.status != 200:
                            print(f"Failed to broadcast to peer {peer}: Status {response.status}")
            except aiohttp.ClientConnectorError as e:
                print(f"Failed to connect to peer {peer} for broadcast: {e}")
