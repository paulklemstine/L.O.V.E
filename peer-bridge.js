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
const DHT = require('bittorrent-dht');

// --- Configuration ---
const RECONNECT_DELAY = 5000; // 5 seconds

// --- State ---
let peer = null;
const connections = new Map();
let isReconnecting = false;
let peerId = `love-client-${uuidv4()}`; // Always generate a unique ID
const dht = new DHT();
const LOVE_INFO_HASH = 'd24713a43a2c2625b682b13f28247078bd440306'; // Info hash for L.O.V.E. network

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

// --- Server Selection Logic ---
const PUBLIC_ICE_SERVERS = [
    'stun:stun.l.google.com:19302',
    'stun:stun1.l.google.com:19302',
    'stun:stun2.l.google.com:19302',
    'stun:stun3.l.google.com:19302',
    'stun:stun4.l.google.com:19302',
    'stun:stun.ekiga.net',
    'stun:stun.ideasip.com',
    'stun:stun.rixtelecom.se',
    'stun:stun.schlund.de',
    'stun:stun.stunprotocol.org:3478',
    'stun:stun.voiparound.com',
    'stun:stun.voipbuster.com',
    'stun:stun.voipstunt.com',
    'stun:stun.voxgratia.org'
];

async function selectOptimalIceServers(servers, numToSelect = 5, testTimeout = 2000) {
    log('info', `Selecting optimal ICE servers from ${servers.length} candidates...`);

    const testServer = async (serverUrl) => {
        const startTime = Date.now();
        return new Promise((resolve) => {
            try {
                const pc = new RTCPeerConnection({ iceServers: [{ urls: serverUrl }] });

                // Set up a timeout for the test
                const timeoutId = setTimeout(() => {
                    resolve(null); // Resolve with null on timeout
                    pc.close();
                }, testTimeout);

                pc.onicecandidate = (event) => {
                    if (event.candidate) {
                        const latency = Date.now() - startTime;
                        clearTimeout(timeoutId);
                        resolve({ server: serverUrl, latency });
                        pc.close();
                    }
                };

                // Create a dummy data channel to trigger ICE candidate gathering
                pc.createDataChannel('latency-test');
                pc.createOffer().then(offer => pc.setLocalDescription(offer));

            } catch (error) {
                log('warn', `Error testing STUN server ${serverUrl}: ${error.message}`);
                resolve(null);
            }
        });
    };

    const results = await Promise.all(servers.map(testServer));
    const successfulServers = results.filter(result => result !== null);

    successfulServers.sort((a, b) => a.latency - b.latency);

    const bestServers = successfulServers.slice(0, numToSelect).map(result => ({ urls: result.server }));

    if (bestServers.length === 0) {
        log('warn', 'No responsive STUN servers found. Using default Google servers as fallback.');
        return [{ urls: 'stun:stun.l.google.com:19302' }];
    }

    log('info', `Selected ${bestServers.length} fastest servers: ${JSON.stringify(bestServers.map(s => s.urls))}`);
    return bestServers;
}


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
function connectToPeer(targetPeerId, options = {}) {
    if (!peer || peer.destroyed) {
        log('error', 'Cannot connect to peer, main peer object is not initialized.');
        return;
    }
    if (connections.has(targetPeerId) || targetPeerId === peerId) {
        log('info', `Already connected to ${targetPeerId} or it is self, skipping.`);
        return;
    }

    const { reliable = true } = options;
    log('info', `Attempting to establish direct connection to peer: ${targetPeerId} (reliable: ${reliable})`);
    const conn = peer.connect(targetPeerId, { reliable });

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
    });

    conn.on('data', (data) => {
        // Handle regular data messages
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


async function initializePeer() {
    // Use the globally stored peerId
    if (peer) {
        log('warn', 'Peer already initialized.');
        return;
    }

    const optimalServers = await selectOptimalIceServers(PUBLIC_ICE_SERVERS);


    const peerConfig = {
        wrtc: wrtc,
        config: {
            'iceServers': optimalServers
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

        // Signal to Python that we are online.
        process.stdout.write(JSON.stringify({ type: 'status', status: 'online', peerId: id }) + '\n');

        // Announce ourselves on the DHT and look for other peers.
        dht.announce(LOVE_INFO_HASH, peer.options.port, (err) => {
            if (err) log('error', `DHT announce error: ${err.message}`);
            log('info', 'Announced self on the L.O.V.E. DHT infohash.');
        });
        dht.lookup(LOVE_INFO_HASH, (err) => {
            if (err) log('error', `DHT lookup error: ${err.message}`);
            log('info', 'Looking up peers on the L.O.V.E. DHT infohash.');
        });
    });

    peer.on('connection', (conn) => {
        handleNewConnection(conn);
    });

    peer.on('error', (err) => {
        // For all other errors, log them and decide whether to reconnect.
        log('error', `PeerJS error: ${err.type} - ${err.message}`);
        {
            const recoverableErrors = ['network', 'server-error', 'socket-error', 'peer-unavailable', 'webrtc', 'unavailable-id'];
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
            connectToPeer(message.peerId, { reliable: message.reliable });
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
async function main() {
    log('info', "Starting PeerJS Node.js bridge...");

    dht.listen(20000, () => {
        log('info', 'DHT is listening on port 20000.');
    });

    dht.on('peer', (peer, infoHash, from) => {
        log('info', `Found potential peer ${peer.host}:${peer.port} through ${from.address}:${from.port}`);
        // The peer object from bittorrent-dht is just host/port. We need a PeerJS ID to connect.
        // For now, we'll assume the PeerJS ID is the same as the host:port combination.
        // This is a simplification and might need to be revisited.
        const discoveredPeerId = `${peer.host}:${peer.port}`;
        connectToPeer(discoveredPeerId);
    });

    await initializePeer();
    listenToPython();
}

main().catch(err => {
    log('error', `Unhandled error in main execution: ${err.message}`);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
    log('info', 'Received SIGINT. Shutting down bridge...');
    if (peer) {
        peer.destroy();
    }
    process.exit(0);
});