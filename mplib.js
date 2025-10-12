const MPLib = (() => {
    // --- Overall State ---
    const API_KEY = 'peerjs'; // Public PeerJS API key
    const MASTER_DIRECTORY_ID = 'sparksync-master-directory-v2';
    let localMasterId = null; // This is the user's "permanent" ID
    let config = {};

    // --- Master Connection State ---
    let masterPeer = null;
    let masterConnection = null; // For clients, the connection to the master
    let isMasterDirectory = false; // Is this client THE master?
    const masterClientConnections = new Map(); // For the master, connections to all clients
    const globalDirectory = { rooms: {}, peers: {} }; // The master's source of truth

    // --- Room Connection State ---
    let roomPeer = null;
    let localRoomId = null;
    let gossipInterval = null;
    const roomConnections = new Map();
    const pendingRoomConnections = new Set();
    const roomKnownPeerIds = new Set();

    // --- Evolve Peer Connection State ---
    let evolvePyPeerConnection = null;

    // --- Callbacks & Config ---
    const defaultConfig = {
        debugLevel: 0,
        onStatusUpdate: (msg, type) => console.log(`[MPLib] ${msg}`),
        onError: (type, err) => console.error(`[MPLib] Error (${type}):`, err),
        onMasterConnected: (id) => {},
        onMasterDisconnected: () => {},
        onNewMasterEstablished: () => {}, // Called when connection to a master is opened
        onDirectoryUpdate: (directory) => {},
        onRoomPeerJoined: (peerId, conn) => {},
        onRoomPeerLeft: (peerId) => {},
        onRoomPeerDisconnected: (masterId) => {}, // For cleaning up game state
        onRoomDataReceived: (peerId, data) => {},
        onRoomConnected: (id) => {},
        onRoomPeerReady: (peerId) => {},
        onPinResponse: (data) => {}, // Callback for when evolve.py returns a pin result
    };

    function logMessage(message, type = 'info') {
        config.onStatusUpdate(message, type);
    }

    // --- Initialization ---

    /**
     * Initializes the entire library, starting the connection to the master directory.
     */
    function initialize(options = {}) {
        config = { ...defaultConfig, ...options };
        logMessage("Initializing MPLib and connecting to Master Directory...", 'info');
        connectToMasterDirectory();
    }

    // --- Master Directory Functions ---

    function connectToMasterDirectory() {
        // First, try to BECOME the master directory
        logMessage(`Attempting to become Master Directory with ID: ${MASTER_DIRECTORY_ID}`, 'info');
        masterPeer = new Peer(MASTER_DIRECTORY_ID, { debug: config.debugLevel, key: API_KEY });
        setupMasterPeerListeners();

        masterPeer.on('open', (id) => {
            // If we get here with the master ID, we ARE the master directory
            localMasterId = id;
            isMasterDirectory = true;
            logMessage(`*** THIS CLIENT IS THE MASTER DIRECTORY (ID: ${id}) ***`, 'success');
            config.onMasterConnected(id);
            // Also trigger the "established" callback for the master itself, so it joins the room.
            config.onNewMasterEstablished();
        });

        masterPeer.on('error', (err) => {
            if (err.type === 'unavailable-id') {
                // The master directory already exists. Destroy this peer and connect as a client.
                logMessage(`Master Directory already exists. Connecting as client.`, 'info');
                if (masterPeer && !masterPeer.destroyed) {
                    masterPeer.destroy();
                }
                connectAsClient();
            } else {
                config.onError('master-peer', err);
            }
        });
    }

    function connectAsClient() {
        isMasterDirectory = false;
        masterPeer = new Peer({ debug: config.debugLevel, key: API_KEY });

        setupMasterPeerListeners(); // General listeners for disconnect, etc.

        // As a client, we should not accept incoming connections on the master peer.
        masterPeer.on('connection', (conn) => {
            logMessage(`Client received unexpected connection from ${conn.peer}. Rejecting.`, 'warn');
            conn.close();
        });

        masterPeer.on('open', (id) => {
            localMasterId = id;
            logMessage(`Connected to signaling server with client master ID: ${id}`, 'info');
            config.onMasterConnected(id);
            // Now, connect to the actual master directory
            masterConnection = masterPeer.connect(MASTER_DIRECTORY_ID, { reliable: true });
            setupMasterClientConnection(masterConnection);
        });

        masterPeer.on('error', (err) => {
            config.onError('client-peer', err);
        });
    }

    function broadcastDirectoryUpdate() {
        if (!isMasterDirectory) return;
        logMessage(`Broadcasting directory update to ${masterClientConnections.size} clients.`, 'info');
        const payload = { type: 'directory-update', payload: globalDirectory };
        masterClientConnections.forEach(conn => {
            if (conn && conn.open) {
                conn.send(payload);
            }
        });
    }

    function setupMasterPeerListeners() {
        masterPeer.on('connection', (conn) => {
            if (isMasterDirectory) {
                logMessage(`Master Directory: New client connected ${conn.peer}`, 'info');
                masterClientConnections.set(conn.peer, conn);

                // When the connection is established, send the new client the current directory
                conn.on('open', () => {
                    if (conn.open) {
                        conn.send({ type: 'directory-update', payload: globalDirectory });
                    }
                });

                conn.on('data', (data) => {
                    if (!isMasterDirectory) return;
                    const peerId = conn.peer;

                    if (data.type === 'update_status') {
                        const { newRoom, isPublic } = data.payload;

                        // 1. Remove peer from their old room in the directory
                        const oldRoomName = globalDirectory.peers[peerId]?.currentRoom;
                        if (oldRoomName && globalDirectory.rooms[oldRoomName]) {
                            globalDirectory.rooms[oldRoomName].occupants =
                                globalDirectory.rooms[oldRoomName].occupants.filter(p => p !== peerId);
                            // Clean up the room if it's now empty
                            if (globalDirectory.rooms[oldRoomName].occupants.length === 0) {
                                delete globalDirectory.rooms[oldRoomName];
                            }
                        }

                        // 2. Add peer to their new room if it's public
                        if (newRoom && isPublic) {
                            if (!globalDirectory.rooms[newRoom]) {
                                globalDirectory.rooms[newRoom] = { occupants: [] };
                            }
                            // Avoid duplicate entries
                            if (!globalDirectory.rooms[newRoom].occupants.includes(peerId)) {
                                globalDirectory.rooms[newRoom].occupants.push(peerId);
                            }
                            globalDirectory.peers[peerId] = { currentRoom: newRoom };
                        } else {
                            // If the room is private or the user is leaving, just remove their entry
                            delete globalDirectory.peers[peerId];
                        }

                        logMessage(`Updated directory for peer ${peerId}, now in room: ${newRoom || 'none'}`);
                        broadcastDirectoryUpdate(); // Inform everyone of the change
                    }
                });

                conn.on('close', () => {
                    logMessage(`Master Directory: Client disconnected ${conn.peer}`, 'warn');
                    const peerId = conn.peer;
                    masterClientConnections.delete(peerId);

                    // Clean up directory
                    const oldRoomName = globalDirectory.peers[peerId]?.currentRoom;
                    if (oldRoomName && globalDirectory.rooms[oldRoomName]) {
                        globalDirectory.rooms[oldRoomName].occupants =
                            globalDirectory.rooms[oldRoomName].occupants.filter(p => p !== peerId);
                        if (globalDirectory.rooms[oldRoomName].occupants.length === 0) {
                            delete globalDirectory.rooms[oldRoomName];
                        }
                    }
                    delete globalDirectory.peers[peerId];

                    logMessage(`Cleaned up directory for disconnected peer ${peerId}`);
                    broadcastDirectoryUpdate(); // Inform everyone
                });

            } else {
                logMessage(`Client received unexpected connection from ${conn.peer}`, 'warn');
                conn.close();
            }
        });

        masterPeer.on('disconnected', () => {
            logMessage('Disconnected from master signaling server. Attempting to reconnect...', 'warn');
            masterPeer.reconnect();
        });

        masterPeer.on('close', () => {
             logMessage('Master peer connection fully closed.', 'error');
             config.onMasterDisconnected();
        });
    }

    function setupMasterClientConnection(conn) {
        conn.on('open', () => {
            logMessage(`Connection to master peer ${conn.peer} is open.`, 'success');
            // Notify the main script that we have a master and it should report its status.
            config.onNewMasterEstablished();
        });

        conn.on('data', (data) => {
            // This is where a client receives data FROM the master
            if (data.type === 'directory-update') {
                config.onDirectoryUpdate(data.payload);
            } else {
                 config.onRoomDataReceived(conn.peer, data); // Fallback for now
            }
        });
        conn.on('close', () => {
            logMessage('Connection to master directory closed. Triggering re-election...', 'error');
            masterConnection = null;
            config.onMasterDisconnected();
            // Wait a short, random amount of time to prevent all clients from trying to reconnect at the exact same moment
            setTimeout(() => {
                connectToMasterDirectory();
            }, Math.random() * 1500 + 500); // Random delay between 0.5s and 2s
        });
         conn.on('error', (err) => {
            logMessage(`Error with master directory connection: ${err.message}`, 'error');
        });
    }

    // --- Room Functions ---

    function joinRoom(roomName) {
        if (!roomName) {
            config.onError('join-room', 'Room Name is required.');
            return;
        }
        if (roomPeer && !roomPeer.destroyed) {
            logMessage(`Already in a room. Disconnecting from previous room first.`, 'warn');
            leaveRoom();
        }

        const seedId = `sparksync-lobby-${roomName.replace(/[^a-zA-Z0-9-]/g, '-')}`;
        logMessage(`Attempting to join room '${roomName}' with seed ID: ${seedId}`, 'info');
        roomPeer = new Peer(seedId, { debug: config.debugLevel, key: API_KEY });
        setupRoomPeerListeners(seedId);

        // Start the proactive gossip interval
        if (gossipInterval) clearInterval(gossipInterval);
        gossipInterval = setInterval(() => {
            const peers = Array.from(roomConnections.keys());
            if (peers.length > 0) {
                const randomPeerId = peers[Math.floor(Math.random() * peers.length)];
                const conn = roomConnections.get(randomPeerId);
                if (conn && conn.open) {
                    logMessage(`Proactive gossip: requesting peer list from ${randomPeerId.slice(-6)}`, 'info');
                    conn.send({ type: 'request-peer-list' });
                }
            }
        }, 7500); // Gossip every 7.5 seconds
    }

    function leaveRoom() {
        if (gossipInterval) {
            clearInterval(gossipInterval);
            gossipInterval = null;
        }
        if(roomPeer) {
            roomPeer.destroy();
            roomPeer = null;
            localRoomId = null;
            roomConnections.clear();
            pendingRoomConnections.clear();
            roomKnownPeerIds.clear();
            logMessage("Left room and cleaned up connections.", 'info');
        }
    }

    function setupRoomPeerListeners(seedId) {
        roomPeer.on('open', (id) => {
            localRoomId = id;
            roomKnownPeerIds.add(id);
            config.onRoomConnected(id);
            // If our ID is the seedId, we are the seed. Otherwise, connect to the seed.
            if (id !== seedId) {
                logMessage(`Connected to room with anonymous ID: ${id}. Now connecting to seed.`, 'info');
                connectToRoomPeer(seedId);
            } else {
                 logMessage(`Successfully registered as room seed with ID: ${id}`, 'success');
            }
        });

        roomPeer.on('connection', (conn) => {
            logMessage(`Incoming room connection from ${conn.peer}`, 'info');
            setupRoomConnection(conn);
        });

        roomPeer.on('error', (err) => {
            if (err.type === 'unavailable-id') {
                // This is expected. It means the room exists. Destroy the failed peer and make an anonymous one.
                roomPeer.destroy();
                roomPeer = new Peer({ debug: config.debugLevel, key: API_KEY });
                setupRoomPeerListeners(seedId); // Re-run setup with the same seed target
            } else {
                config.onError('room-peer', err);
            }
        });
    }

    function connectToRoomPeer(targetPeerId) {
        if (!roomPeer || roomPeer.destroyed || targetPeerId === localRoomId || roomConnections.has(targetPeerId) || pendingRoomConnections.has(targetPeerId)) {
            return;
        }
        logMessage(`Attempting to connect to room peer ${targetPeerId}`, 'info');
        pendingRoomConnections.add(targetPeerId);

        const conn = roomPeer.connect(targetPeerId, {
            reliable: true,
            metadata: { masterId: localMasterId } // Share our permanent ID
        });
        conn.on('open', () => {
            logMessage(`Room connection opened with ${targetPeerId}`, 'info');
            pendingRoomConnections.delete(targetPeerId);
            setupRoomConnection(conn);
        });
        conn.on('error', (err) => {
            logMessage(`Room connection error with ${targetPeerId}: ${err.message}`, 'error');
            pendingRoomConnections.delete(targetPeerId);
        });
    }

    function setupRoomConnection(conn) {
        const remotePeerId = conn.peer;
        conn.ready = false; // Add a 'ready' flag to the connection object
        roomConnections.set(remotePeerId, conn);
        roomKnownPeerIds.add(remotePeerId);

        const onConnectionOpen = () => {
            logMessage(`Connection with ${remotePeerId.slice(-6)} is open. Sending ready signal.`, 'info');
            conn.send({ type: 'peer_ready' });
            config.onRoomPeerJoined(remotePeerId, conn);
        };

        if (conn.open) {
            onConnectionOpen();
        } else {
            conn.on('open', onConnectionOpen);
        }

        conn.on('data', (data) => {
            switch (data.type) {
                case 'peer_ready':
                    if (!conn.ready) {
                        logMessage(`Peer ${remotePeerId.slice(-6)} is ready.`, 'success');
                        conn.ready = true;
                        config.onRoomPeerReady(remotePeerId);
                        // Now that they are ready, ask for their peers
                        conn.send({ type: 'request-peer-list' });
                    }
                    break;
                case 'request-peer-list':
                    conn.send({ type: 'peer-list-update', list: Array.from(roomKnownPeerIds) });
                    break;
                case 'peer-list-update':
                    data.list.forEach(id => {
                        if (id !== localRoomId && !roomConnections.has(id) && !pendingRoomConnections.has(id)) {
                            connectToRoomPeer(id);
                        }
                    });
                    break;
                default:
                    // Pass any other data types to the main application
                    config.onRoomDataReceived(remotePeerId, data);
                    break;
            }
        });

        conn.on('close', () => {
            logMessage(`Room connection closed with ${remotePeerId}`, 'warn');
            removeRoomConnection(remotePeerId);
        });
    }

    function removeRoomConnection(peerId) {
        const conn = roomConnections.get(peerId);
        if (conn) {
            // Get the persistent masterId before deleting the connection
            const masterId = conn.metadata?.masterId;

            roomConnections.delete(peerId);
            roomKnownPeerIds.delete(peerId);
            logMessage(`Removed room connection for ${peerId}`, 'info');
            config.onRoomPeerLeft(peerId); // For general UI updates

            // Use the new callback for specific game state cleanup
            if (masterId) {
                config.onRoomPeerDisconnected(masterId);
            }
        }
    }

    function broadcastToRoom(payload, peerIds = null) {
        const targets = peerIds ? peerIds : Array.from(roomConnections.keys());
        logMessage(`Broadcasting payload of type ${payload.type} to ${targets.length} peers.`, 'info');
        targets.forEach(peerId => {
            const conn = roomConnections.get(peerId);
            if (conn && conn.open) {
                conn.send(payload);
            }
        });
    }

    function sendToMaster(payload) {
        if (masterConnection && masterConnection.open) {
            masterConnection.send(payload);
        } else {
            logMessage("Cannot send to master: No open connection.", 'warn');
        }
    }

    function sendDirectToRoomPeer(targetPeerId, payload) {
        const conn = roomConnections.get(targetPeerId);
        if (conn && conn.open) {
            try {
                conn.send(payload);
            } catch (e) {
                logMessage(`Error sending direct message to room peer ${targetPeerId}: ${e.message}`, 'error');
            }
        } else {
            logMessage(`No open room connection to ${targetPeerId} for direct message.`, 'warn');
        }
    }

    // --- Public API ---
    const publicApi = {
        initialize,
        joinRoom,
        leaveRoom,
        broadcastToRoom,
        sendDirectToRoomPeer,
        sendToMaster,
        closeConnection: (peerId) => { // New function
            const conn = roomConnections.get(peerId);
            if (conn) {
                logMessage(`Manually closing connection to ${peerId}`, 'warn');
                conn.close();
                // The 'close' event handler on the connection will call removeRoomConnection
            }
        },
        getLocalMasterId: () => localMasterId,
        getLocalRoomId: () => localRoomId,
        getRoomConnections: () => new Map(roomConnections),
        getRoomKnownPeerIds: () => new Set(roomKnownPeerIds),
        connectToEvolvePeer,
        sendCodeToEvolvePeer,
    };

    // --- Evolve Peer Functions ---

    /**
     * Connects to a specific evolve.py peer instance.
     * This is separate from the room logic.
     */
    function connectToEvolvePeer(targetPeerId) {
        if (!masterPeer || masterPeer.destroyed) {
            logMessage('Cannot connect to evolve peer: Master peer is not initialized.', 'error');
            return;
        }
        if (evolvePyPeerConnection && evolvePyPeerConnection.open) {
            logMessage(`Already connected to an evolve peer: ${evolvePyPeerConnection.peer}. Disconnecting first.`, 'warn');
            evolvePyPeerConnection.close();
        }

        logMessage(`Attempting to connect to evolve.py peer: ${targetPeerId}`, 'info');
        const conn = masterPeer.connect(targetPeerId, { reliable: true });

        conn.on('open', () => {
            logMessage(`Successfully connected to evolve.py peer: ${conn.peer}`, 'success');
            evolvePyPeerConnection = conn;
            // Setup listeners for this specific connection
            setupEvolvePeerConnection(conn);
        });

        conn.on('error', (err) => {
            logMessage(`Failed to connect to evolve.py peer ${targetPeerId}: ${err.message}`, 'error');
        });
    }

    /**
     * Sets up the data and close handlers for the connection to the evolve.py peer.
     */
    function setupEvolvePeerConnection(conn) {
        conn.on('data', (data) => {
            // Data received from the evolve.py peer
            logMessage(`Received data from evolve.py peer: ${JSON.stringify(data)}`, 'info');
            if (data.type === 'pin-response') {
                // Pass the response to the main application via callback
                config.onPinResponse(data);
            }
        });

        conn.on('close', () => {
            logMessage(`Connection to evolve.py peer ${conn.peer} closed.`, 'warn');
            if (evolvePyPeerConnection && evolvePyPeerConnection.peer === conn.peer) {
                evolvePyPeerConnection = null;
            }
        });
    }

    /**
     * Sends the provided source code to the connected evolve.py peer.
     */
    function sendCodeToEvolvePeer(sourceCode) {
        if (evolvePyPeerConnection && evolvePyPeerConnection.open) {
            logMessage('Sending source code to evolve.py peer for IPFS pinning...', 'info');
            evolvePyPeerConnection.send(sourceCode);
        } else {
            logMessage('Cannot send source code: Not connected to an evolve.py peer.', 'error');
            config.onError('send-code', 'Not connected to an evolve.py peer.');
        }
    }


    return publicApi;
})();

export default MPLib;
