const { Peer } = require('@its-forked/peerjs-on-node');
const readline = require('readline');

// --- Constants ---
const API_KEY = 'peerjs';
const LOBBY_ID = 'borg-lobby';

// --- State ---
let peer = null;
const connections = new Map(); // Stores direct connections from browser peers

function logToPython(message, level = 'info') {
    const logObject = { type: 'log', level, message };
    console.error(JSON.stringify(logObject));
}

logToPython('PeerJS bridge starting...', 'info');

function initializePeer() {
    peer = new Peer(LOBBY_ID, { key: API_KEY });

    peer.on('open', (id) => {
        logToPython(`Lobby is online with ID: ${id}`, 'success');
        console.log(JSON.stringify({ type: 'status', status: 'online', peerId: id }));
    });

    peer.on('connection', (conn) => {
        logToPython(`Peer connected to lobby: ${conn.peer}`);
        connections.set(conn.peer, conn);

        // Announce capability to the newly connected peer
        conn.on('open', () => {
            conn.send({
                type: 'capability-announcement',
                payload: { capability: 'evolve.py' }
            });
        });

        conn.on('data', (data) => {
            if(data.type === 'pin-request') {
                logToPython(`Pin request received from ${conn.peer}`);
                const messageToPython = {
                    type: 'pin-request',
                    peerId: conn.peer,
                    payload: data.payload
                };
                console.log(JSON.stringify(messageToPython));
            }
        });

        conn.on('close', () => {
            logToPython(`Peer disconnected from lobby: ${conn.peer}`);
            connections.delete(conn.peer);
        });
    });

    peer.on('error', (err) => {
        // If the ID is taken, it's a critical error because we can't be the lobby.
        logToPython(`Fatal peer error: ${err.message}`, 'error');
        console.error(JSON.stringify({ type: 'status', status: 'error', message: err.message }));
        process.exit(1);
    });
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

rl.on('line', (line) => {
    try {
        const command = JSON.parse(line);
        if (command.type === 'send-response' && command.peerId && command.payload) {
            const targetConn = connections.get(command.peerId);
            if (targetConn && targetConn.open) {
                targetConn.send(command.payload);
                logToPython(`Relayed response to peer: ${command.peerId}`);
            } else {
                logToPython(`Could not send response. Peer not found or connection closed: ${command.peerId}`, 'error');
            }
        }
    } catch (error) {
        logToPython(`Error parsing command from Python: ${error.message}`, 'error');
    }
});

process.on('SIGINT', () => {
    logToPython('PeerJS bridge shutting down...');
    if (peer) peer.destroy();
    process.exit(0);
});

initializePeer();

setInterval(() => {}, 1000 * 60);