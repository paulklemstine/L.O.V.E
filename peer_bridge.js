const { Peer } = require('@its-forked/peerjs-on-node');
const readline = require('readline');

// This script acts as a bridge between the Python application (evolve.py)
// and the P2P network, allowing Python to interact with browser-based peers.

console.log('PeerJS bridge starting...');

const peer = new Peer();
const connections = new Map(); // Store active connections by peer ID

// 1. PeerJS Event Handling
// ------------------------

peer.on('open', (id) => {
    // Notify Python that the bridge is online and what its Peer ID is.
    // Python will capture this from stdout.
    console.log(JSON.stringify({ type: 'status', status: 'online', peerId: id }));
});

peer.on('error', (err) => {
    // Notify Python of any fatal errors.
    console.error(JSON.stringify({ type: 'status', status: 'error', message: err.message }));
    process.exit(1);
});

peer.on('connection', (conn) => {
    // A new browser peer has connected to us.
    logToPython(`Connection established with peer: ${conn.peer}`);
    connections.set(conn.peer, conn); // Store the connection object

    conn.on('data', (data) => {
        // Data received from a browser peer (e.g., a request to pin source code).
        // We forward this data to the Python script for processing.
        logToPython(`Data received from ${conn.peer}`);
        const messageToPython = {
            type: 'pin-request',
            peerId: conn.peer,
            payload: data // The data is expected to be the HTML source code
        };
        // The Python script will be listening to stdout for this JSON.
        console.log(JSON.stringify(messageToPython));
    });

    conn.on('close', () => {
        // The connection was closed.
        logToPython(`Connection closed with peer: ${conn.peer}`);
        connections.delete(conn.peer);
    });

    conn.on('error', (err) => {
        // An error occurred on this specific connection.
        logToPython(`Connection error with ${conn.peer}: ${err.message}`, 'error');
        connections.delete(conn.peer); // Clean up on error
    });
});

// 2. Python Communication Handling
// --------------------------------

// Create an interface to read commands from Python's stdin.
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

// Listen for lines from the Python script.
rl.on('line', (line) => {
    try {
        const command = JSON.parse(line);
        logToPython(`Received command from Python: ${command.type}`);

        if (command.type === 'send-response' && command.peerId && command.payload) {
            const targetConn = connections.get(command.peerId);
            if (targetConn && targetConn.open) {
                // We found the target peer's connection, so send the payload.
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

// 3. Utility and Cleanup
// ----------------------

function logToPython(message, level = 'info') {
    // A structured logging format for Python to distinguish logs from data.
    const logObject = {
        type: 'log',
        level: level,
        message: message
    };
    // Use console.error for logs to keep stdout clean for primary data.
    console.error(JSON.stringify(logObject));
}

// Gracefully shut down on SIGINT (Ctrl+C).
process.on('SIGINT', () => {
    logToPython('PeerJS bridge shutting down...');
    peer.destroy();
    process.exit(0);
});

// Keep the process alive.
setInterval(() => {}, 1000 * 60);