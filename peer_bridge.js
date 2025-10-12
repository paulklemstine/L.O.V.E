const { Peer } = require('@its-forked/peerjs-on-node');

console.log('Starting PeerJS bridge...');

const peer = new Peer();

peer.on('open', (id) => {
    console.log('PeerJS bridge connected with ID:', id);
});

peer.on('error', (err) => {
    console.error('PeerJS bridge error:', err);
    process.exit(1);
});

peer.on('connection', (conn) => {
    console.log('PeerJS bridge connection from:', conn.peer);
    conn.on('data', (data) => {
        console.log('PeerJS bridge received data:', data);
    });
});

// Keep the process alive
setInterval(() => {}, 1000);

process.on('SIGINT', () => {
    console.log('PeerJS bridge shutting down...');
    peer.destroy();
    process.exit(0);
});