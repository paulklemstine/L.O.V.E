// --- Shim for Node.js ---
const wrtc = require('@roamhq/wrtc');
global.RTCPeerConnection = wrtc.RTCPeerConnection;
global.RTCSessionDescription = wrtc.RTCSessionDescription;
global.RTCIceCandidate = wrtc.RTCIceCandidate;
global.navigator = { userAgent: 'Node.js' };

const { Peer } = require('peerjs');
const { Readable } = require('stream');

// --- Configuration ---
const PEER_ID_PREFIX = 'evolve-node-';
const STUN_SERVERS = [
    'stun:stun.l.google.com:19302',
    'stun:global.stun.twilio.com:3478'
];

// --- State ---
let peer = null;
const connections = new Map();

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

// --- PeerJS Logic ---
function initializePeer() {
    // [Jules] Changed to a static ID to create a discoverable lobby.
    const peerId = 'borg-lobby';

    peer = new Peer(peerId, {
        wrtc: wrtc,
        config: {
            'iceServers': [{ urls: STUN_SERVERS }]
        },
        // Use a more robust connection timeout
        connectTimeout: 10000
    });

    peer.on('open', (id) => {
        log('info', `PeerJS connection opened with ID: ${id}`);
        // Signal to Python that the bridge is ready.
        // NOTE: The format is flattened (no 'payload' object) to match the
        // parsing logic in the parent network.py script.
        process.stdout.write(JSON.stringify({ type: 'status', status: 'online', peerId: id }) + '\n');
    });

    peer.on('connection', (conn) => {
        log('info', `Received connection from ${conn.peer}`);
        connections.set(conn.peer, conn);
        process.stdout.write(JSON.stringify({ type: 'connection', peer: conn.peer }) + '\n');

        conn.on('data', (data) => {
            log('info', `Received data from ${conn.peer}`);

            // Forward the data to the Python script via stdout
            // Add the peer ID so Python knows who to reply to
            const messageToPython = {
                type: 'p2p-data',
                peer: conn.peer,
                payload: data
            };
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
    });

    peer.on('error', (err) => {
        log('error', `PeerJS error: ${err.type} - ${err.message}`);
        // Signal a critical error to Python.
        // NOTE: The format is flattened (no 'payload' object) to match the
        // parsing logic in the parent network.py script.
        process.stdout.write(JSON.stringify({ type: 'status', status: 'error', message: err.message }) + '\n');
        process.exit(1);
    });

    peer.on('disconnected', () => {
        log('warn', 'Peer disconnected from signaling server. Attempting to reconnect...');
        // PeerJS will automatically try to reconnect.
    });

    peer.on('close', () => {
        log('error', 'Peer connection closed permanently.');
        process.exit(1);
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