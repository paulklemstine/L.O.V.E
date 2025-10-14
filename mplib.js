const MPLib = (() => {
    // --- Overall State ---
    const API_KEY = 'peerjs'; // Public PeerJS API key
    const BORG_LOBBY_ID = 'borg-lobby';
    let config = {};

    // --- Peer Connection State ---
    let peer = null;
    let localPeerId = null;
    let isHost = false;
    const connections = new Map(); // For host to manage clients, or for client to store its host connection

    // --- Callbacks & Config ---
    const defaultConfig = {
        debugLevel: 0,
        onStatusUpdate: (msg, type) => console.log(`[MPLib] ${msg}`),
        onError: (type, err) => console.error(`[MPLib] Error (${type}):`, err),
        onConnectionChange: (count) => {}, // New callback for connection count changes
        onRoomDataReceived: (peerId, data) => {},
    };

    function logMessage(message, type = 'info') {
        config.onStatusUpdate(message, type);
    }

    // --- Initialization ---
    function initialize(options = {}) {
        config = { ...defaultConfig, ...options };
        logMessage("Initializing MPLib...", 'info');
        tryToBecomeHost();
    }

    function tryToBecomeHost() {
        logMessage(`Attempting to become the lobby host: ${BORG_LOBBY_ID}`, 'info');
        peer = new Peer(BORG_LOBBY_ID, { debug: config.debugLevel, key: API_KEY });

        peer.on('open', id => {
            isHost = true;
            localPeerId = id;
            logMessage(`Lobby Host`, 'status'); // Simplified status message
            config.onStatusUpdate(`Successfully became the lobby host with ID: ${id}`, 'info');
            setupHostListeners();
        });

        peer.on('error', err => {
            if (err.type === 'unavailable-id') {
                config.onStatusUpdate(`Lobby host already exists. Connecting as a client...`, 'info');
                peer.destroy();
                connectAsClient();
            } else {
                config.onError('peer-error', err);
            }
        });
    }

    function connectAsClient() {
        peer = new Peer({ debug: config.debugLevel, key: API_KEY });
        isHost = false;

        peer.on('open', id => {
            localPeerId = id;
            config.onStatusUpdate(`Peer ID assigned: ${id}.`, 'info');
            connectToLobby();
        });

        peer.on('error', (err) => {
            config.onError('peer-error', err);
        });
    }

    function setupHostListeners() {
        config.onStatusUpdate('Lobby is online. Waiting for connections...', 'info');
        peer.on('connection', conn => {
            config.onStatusUpdate(`Incoming connection from ${conn.peer}`, 'info');
            connections.set(conn.peer, conn);
            config.onConnectionChange(connections.size);

            conn.on('data', data => {
                config.onStatusUpdate(`Host received data from ${conn.peer}, broadcasting...`, 'info');
                // Broadcast data to all other clients
                for (const [peerId, connection] of connections.entries()) {
                    if (peerId !== conn.peer) {
                        connection.send(data);
                    }
                }
                // Also, let the host application process the data
                config.onRoomDataReceived(conn.peer, data);
            });

            conn.on('close', () => {
                config.onStatusUpdate(`Connection from ${conn.peer} closed.`, 'warn');
                connections.delete(conn.peer);
                config.onConnectionChange(connections.size);
            });
        });
    }

    function connectToLobby() {
        config.onStatusUpdate(`Attempting to connect to lobby peer: ${BORG_LOBBY_ID}`, 'info');
        const conn = peer.connect(BORG_LOBBY_ID, { reliable: true });

        conn.on('open', () => {
            logMessage(`Successfully connected to lobby`, 'status');
            connections.set(conn.peer, conn); // Store the single connection to the host
            config.onConnectionChange(connections.size);
            setupClientConnection(conn);
        });

        conn.on('error', (err) => {
            logMessage(`Failed to connect to lobby: ${err.message}`, 'error');
        });
    }

    function setupClientConnection(conn) {
        conn.on('data', (data) => {
            config.onRoomDataReceived(conn.peer, data);
        });

        conn.on('close', () => {
            logMessage(`Connection to lobby closed.`, 'warn');
            connections.delete(conn.peer);
            config.onConnectionChange(connections.size);
        });
    }

    function broadcastToRoom(data) {
        if (connections.size === 0) {
            logMessage('No connections available to send data.', 'error');
            return;
        }

        for (const connection of connections.values()) {
            if (connection.open) {
                connection.send(data);
            }
        }
    }

    function sendCodeToEvolvePeer(sourceCode) {
        logMessage('Sending source code to evolve.py peer for IPFS pinning...', 'info');
        const data = { type: 'pin-request', payload: sourceCode };
        broadcastToRoom(data);
    }

    return {
        initialize,
        sendCodeToEvolvePeer,
        broadcastToRoom,
        sendDirectToRoomPeer: (peerId, data) => {
            const conn = connections.get(peerId);
            if (conn && conn.open) {
                conn.send(data);
            } else {
                logMessage(`Could not send direct message to ${peerId}, connection not found or not open.`, 'error');
            }
        },
        getLocalPeerId: () => localPeerId,
        isHost: () => isHost,
        getConnectionCount: () => connections.size,
    };
})();

export default MPLib;