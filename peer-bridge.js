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

// --- Configuration ---
const RECONNECT_DELAY = 5000; // 5 seconds

// --- State ---
let peer = null;
const connections = new Map();
let isReconnecting = false;
// [Jules] Changed to a static ID to create a discoverable lobby.
// Keep the peerId consistent across reconnections.
let peerId = 'borg-lobby';

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
    connections.set(conn.peer, conn);
    process.stdout.write(JSON.stringify({ type: 'connection', peer: conn.peer }) + '\n');

    conn.on('data', (data) => {
        log('info', `Received data from ${conn.peer}`);
        const messageToPython = { type: 'p2p-data', peer: conn.peer, payload: data };
        process.stdout.write(JSON.stringify(messageToPython) + '\n');
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

    peer = new Peer(peerId, {
        wrtc: wrtc,
        config: {
            'iceServers': [{ urls: ['stun:stun.l.google.com:19302', 'stun:global.stun.twilio.com:3478'] }]
        },
        // Use a more robust connection timeout
        connectTimeout: 10000
    });

    peer.on('open', (id) => {
        log('info', `PeerJS connection opened with ID: ${id}`);
        // Update the global peerId in case it was dynamically assigned (shouldn't happen with static ID)
        peerId = id;
        // Signal to Python that the bridge is ready.
        process.stdout.write(JSON.stringify({ type: 'status', status: 'online', peerId: id }) + '\n');
    });

    peer.on('connection', (conn) => {
        handleNewConnection(conn);
    });

    peer.on('error', (err) => {
        log('error', `PeerJS error: ${err.type} - ${err.message}`);

        // Define which errors should trigger a reconnect vs. a fatal exit
        const recoverableErrors = ['network', 'server-error', 'socket-error', 'peer-unavailable', 'webrtc'];

        if (recoverableErrors.includes(err.type)) {
            log('warn', 'A recoverable error occurred. Triggering reconnection.');
            // Signal a transient error to Python without exiting
            process.stdout.write(JSON.stringify({ type: 'status', status: 'reconnecting', message: err.message }) + '\n');
            reconnect();
        } else {
            log('error', 'An unrecoverable error occurred. Exiting.');
            // Signal a critical error to Python.
            process.stdout.write(JSON.stringify({ type: 'status', status: 'error', message: err.message }) + '\n');
            process.exit(1);
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