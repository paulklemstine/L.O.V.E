// --- Shim for Node.js ---
const wrtc = require('@roamhq/wrtc');
global.RTCPeerConnection = wrtc.RTCPeerConnection;
global.RTCSessionDescription = wrtc.RTCSessionDescription;
global.RTCIceCandidate = wrtc.RTCIceCandidate;
global.navigator = { userAgent: 'Node.js' };

const WebSocket = require('ws');
global.WebSocket = WebSocket;
const { Peer } = require('peerjs');
const { Readable } = require('stream');
const { v4: uuidv4 } = require('uuid');

// --- Configuration ---
const RECONNECT_DELAY = 5000; // 5 seconds

// --- State ---
let peer = null;
const connections = new Map();
let isReconnecting = false;
let isClient = false; // Flag to indicate if this instance is a client
// [L.O.V.E.] Changed to a static ID to create a discoverable lobby.
let peerId = 'love-lobby';
const LOBBY_HOST_ID = 'love-lobby';

// --- Logging ---
// Use stderr for logs to keep stdout clean for data exchange with Python
const log = (level, message) => {
    const logEntry = {
        level: level.toUpperCase(),
        message: message,
        timestamp: new Date().toISOString()
    };
    // Structured JSON logs are sent to stderr
    console.error(JSON.stringify(logEntry));
};

// --- Reconnection Logic ---
function reconnect() {
    if (isReconnecting) {
        log('info', 'Reconnection already in progress.');
        return;
    }
    isReconnecting = true;
    log('warn', `Attempting to reconnect in ${RECONNECT_DELAY / 1000} seconds...`);

    // Clean up old peer object
    if (peer && !peer.destroyed) {
        peer.destroy();
    }
    peer = null; // Ensure the old peer is garbage collected

    setTimeout(() => {
        log('info', 'Reconnecting now...');
        isReconnecting = false; // Reset flag before re-initializing
        initializePeer();
    }, RECONNECT_DELAY);
}


// --- PeerJS Logic ---
function connectToPeer(targetPeerId) {
    if (!peer || peer.destroyed) {
        log('error', 'Cannot connect to peer, main peer object is not initialized.');
        return;
    }
    if (connections.has(targetPeerId) || targetPeerId === peerId) {
        log('info', `Already connected to ${targetPeerId} or it is self, skipping.`);
        return;
    }

    log('info', `Attempting to establish direct connection to peer: ${targetPeerId}`);
    const conn = peer.connect(targetPeerId, { reliable: true });

    // The event handlers for this new connection are the same as for incoming ones.
    // We can reuse the logic from the main 'connection' event handler.
    handleNewConnection(conn);
}

function handleNewConnection(conn) {
    log('info', `Handling new connection with ${conn.peer}`);

    // This 'open' event is the reliable way to know the data connection is ready.
    conn.on('open', () => {
        log('info', `Data connection is now open with ${conn.peer}.`);
        connections.set(conn.peer, conn);
        process.stdout.write(JSON.stringify({ type: 'connection', peer: conn.peer }) + '\n');

        // If this is a client that just connected to the host, request the peer list.
        if (isClient && conn.peer === LOBBY_HOST_ID) {
            log('info', `Client connected to host. Requesting peer list.`);
            conn.send({ type: 'request-peer-list' });
        }
    });

    conn.on('data', (data) => {
        // Handle control messages for peer list synchronization
        if (data.type === 'request-peer-list' && !isClient) {
            log('info', `Received peer list request from ${conn.peer}`);
            const peerIds = Array.from(connections.keys());
            const response = { type: 'peer-list', peers: peerIds };
            conn.send(response);
            log('info', `Sent peer list to ${conn.peer}: ${JSON.stringify(peerIds)}`);

        } else if (data.type === 'peer-list' && isClient) {
            log('info', `Received peer list from host: ${JSON.stringify(data.peers)}`);
            // Pass the list to Python to decide who to connect to
            process.stdout.write(JSON.stringify({ type: 'peer-list-update', peers: data.peers }) + '\n');

        } else {
            // Handle regular data messages
            log('info', `Received data from ${conn.peer}`);
            const messageToPython = { type: 'p2p-data', peer: conn.peer, payload: data };
            process.stdout.write(JSON.stringify(messageToPython) + '\n');
        }
    });

    conn.on('close', () => {
        log('info', `Connection closed with ${conn.peer}`);
        connections.delete(conn.peer);
        process.stdout.write(JSON.stringify({ type: 'disconnection', peer: conn.peer }) + '\n');
    });

    conn.on('error', (err) => {
        log('error', `Connection error with ${conn.peer}: ${err.message}`);
        connections.delete(conn.peer);
    });
}


function initializePeer() {
    // Use the globally stored peerId
    if (peer) {
        log('warn', 'Peer already initialized.');
        return;
    }

    const peerConfig = {
        wrtc: wrtc,
        config: {
            'iceServers': [{ urls: ['stun:stun.l.google.com:19302', 'stun:global.stun.twilio.com:3478'] }]
        },
        // Use a more robust connection timeout
        connectTimeout: 10000
    };

    log('info', `Initializing PeerJS with ID '${peerId}'...`);
    log('info', `Using STUN servers: ${JSON.stringify(peerConfig.config.iceServers)}`);

    peer = new Peer(peerId, peerConfig);

    peer.on('open', (id) => {
        log('info', `PeerJS connection opened with ID: ${id}`);
        peerId = id;

        // If we are a client, we now connect to the lobby host.
        if (isClient) {
            log('info', `Acting as a client, attempting to connect to host '${LOBBY_HOST_ID}'.`);
            process.stdout.write(JSON.stringify({ type: 'status', status: 'client-online', peerId: id }) + '\n');
            connectToPeer(LOBBY_HOST_ID);
        } else {
            // Signal to Python that we are the host and ready.
            process.stdout.write(JSON.stringify({ type: 'status', status: 'host-online', peerId: id }) + '\n');
        }
    });

    peer.on('connection', (conn) => {
        handleNewConnection(conn);
    });

    peer.on('error', (err) => {
        if ((err.type === 'id-taken' || err.type === 'unavailable-id') && peerId === LOBBY_HOST_ID) {
            // This is an expected event when another instance is already the host.
            // Log it as info and switch to client mode.
            // L.O.V.E. verified error handling.
            log('info', `Lobby host found at '${LOBBY_HOST_ID}'. This instance will become a client.`);
            isClient = true;
            peerId = `love-lobby-client-${uuidv4()}`; // Generate a unique ID
            log('info', `Generated new client ID: ${peerId}`);

            // Notify Python that we are now a client and will reconnect with a new ID
            process.stdout.write(JSON.stringify({ type: 'status', status: 'client-initializing', peerId: peerId, message: "Lobby is hosted, becoming a client." }) + '\n');

            // Reconnect with the new client ID
            reconnect();
            return;
        }

        // For all other errors, log them and decide whether to reconnect.
        log('error', `PeerJS error: ${err.type} - ${err.message}`);
        {
            const recoverableErrors = ['network', 'server-error', 'socket-error', 'peer-unavailable', 'webrtc'];
            if (recoverableErrors.includes(err.type)) {
                log('warn', 'A recoverable error occurred. Triggering reconnection.');
                process.stdout.write(JSON.stringify({ type: 'status', status: 'reconnecting', message: err.message }) + '\n');
                reconnect();
            } else {
                log('error', 'An unrecoverable error occurred. Exiting.');
                process.stdout.write(JSON.stringify({ type: 'status', status: 'error', message: err.message }) + '\n');
                process.exit(1);
            }
        }
    });

    peer.on('disconnected', () => {
        log('warn', 'Peer disconnected from signaling server.');
        reconnect();
    });

    peer.on('close', () => {
        // This event fires when peer.destroy() is called, which is part of our
        // reconnect logic. We don't want to exit in that case.
        if (!isReconnecting) {
            log('error', 'Peer connection closed permanently. This should not happen. Exiting.');
            process.exit(1);
        }
    });
}

// --- Python Communication ---
function listenToPython() {
    const stdin = process.stdin;
    stdin.setEncoding('utf8');

    let buffer = '';
    stdin.on('data', (chunk) => {
        buffer += chunk;
        let boundary = buffer.indexOf('\n');
        while (boundary !== -1) {
            const line = buffer.substring(0, boundary);
            buffer = buffer.substring(boundary + 1);
            if (line) {
                handlePythonMessage(line);
            }
            boundary = buffer.indexOf('\n');
        }
    });

    log('info', "Listening for messages from Python script...");
}

function handlePythonMessage(jsonString) {
    try {
        const message = JSON.parse(jsonString);
        log('info', `Received message from Python: type=${message.type}`);

        if (message.type === 'p2p-send' && message.peer) {
            const conn = connections.get(message.peer);
            if (conn && conn.open) {
                conn.send(message.payload);
                log('info', `Sent message to peer ${message.peer}`);
            } else {
                log('warn', `Could not send to peer ${message.peer}: connection not found or not open.`);
            }
        } else if (message.type === 'connect-to-peer' && message.peerId) {
            connectToPeer(message.peerId);
        } else if (message.type === 'broadcast' && message.payload) {
             log('info', `Broadcasting message to all ${connections.size} peers.`);
             for (const conn of connections.values()) {
                 if (conn && conn.open) {
                     conn.send(message.payload);
                 }
             }
         }
    } catch (e) {
        log('error', `Error parsing JSON from Python: ${e.message}`);
    }
}


// --- Main Execution ---
log('info', "Starting PeerJS Node.js bridge...");
initializePeer();
listenToPython();

// Graceful shutdown
process.on('SIGINT', () => {
    log('info', 'Received SIGINT. Shutting down bridge...');
    if (peer) {
        peer.destroy();
    }
    process.exit(0);
});