const MPLib = (() => {
    // --- Overall State ---
    const BORG_LOBBY_ID = 'borg-lobby';
    let config = {};

    // --- Peer Connection State ---
    let peer = null;
    let localPeerId = null;
    let isHost = false;
    const directConnections = new Map();

    // --- Callbacks & Config ---
    const defaultConfig = {
        onStatusUpdate: (msg, type) => console.log(`[MPLib] ${msg}`),
        onError: (type, err) => console.error(`[MPLib] Error (${type}):`, err),
        onConnectionChange: (count) => {},
        onRoomDataReceived: (peerId, data) => {},
    };

    // --- Crypto Logic ---
    const SHARED_SECRET = 'a-very-secret-key-that-should-be-exchanged-securely';

    function encrypt(text) {
        return CryptoJS.AES.encrypt(text, SHARED_SECRET).toString();
    }

    function decrypt(ciphertext) {
        const bytes = CryptoJS.AES.decrypt(ciphertext, SHARED_SECRET);
        return bytes.toString(CryptoJS.enc.Utf8);
    }

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
        peer = new Peer(BORG_LOBBY_ID);
        peer.on('open', id => {
            isHost = true;
            localPeerId = id;
            logMessage(`Lobby Host`, 'status');
            peer.on('connection', handleLobbyConnection);
        });
        peer.on('error', err => {
            if (err.type === 'unavailable-id') {
                peer.destroy();
                connectAsClient();
            } else {
                config.onError('peer-error', err);
            }
        });
    }

    function connectAsClient() {
        peer = new Peer();
        peer.on('open', id => {
            localPeerId = id;
            logMessage(`Peer ID: ${id}`, 'status');
            const lobbyConn = peer.connect(BORG_LOBBY_ID);
            lobbyConn.on('data', (data) => {
                if (data.type === 'welcome') {
                    data.peers.forEach(connectToPeer);
                } else if (data.type === 'peer-connect') {
                    connectToPeer(data.peerId);
                }
            });
            peer.on('connection', handleDirectConnection);
        });
        peer.on('error', (err) => config.onError('peer-error', err));
    }

    function handleLobbyConnection(conn) {
        logMessage(`Discovery connection from ${conn.peer}`, 'info');
        conn.on('open', () => {
            conn.send({ type: 'welcome', peers: Array.from(directConnections.keys()) });
            broadcastToRoom({ type: 'peer-connect', peerId: conn.peer });
            connectToPeer(conn.peer);
        });
    }

    function handleDirectConnection(conn) {
        logMessage(`Direct connection from ${conn.peer}`, 'info');
        directConnections.set(conn.peer, conn);
        config.onConnectionChange(directConnections.size);
        conn.on('data', (data) => {
            try {
                const decrypted = JSON.parse(decrypt(data.payload));
                config.onRoomDataReceived(conn.peer, decrypted);
            } catch (e) {
                logMessage(`Failed to decrypt message from ${conn.peer}`, 'error');
            }
        });
        conn.on('close', () => {
            directConnections.delete(conn.peer);
            config.onConnectionChange(directConnections.size);
        });
    }

    function connectToPeer(peerId) {
        if (peerId === localPeerId || directConnections.has(peerId)) return;
        const conn = peer.connect(peerId);
        conn.on('open', () => handleDirectConnection(conn));
    }

    function broadcastToRoom(data) {
        const encrypted = { type: 'encrypted-message', payload: encrypt(JSON.stringify(data)) };
        for (const conn of directConnections.values()) conn.send(encrypted);
    }

    function sendDirectToRoomPeer(peerId, data) {
        const conn = directConnections.get(peerId);
        if (conn) conn.send({ type: 'encrypted-message', payload: encrypt(JSON.stringify(data)) });
    }

    return {
        initialize,
        broadcastToRoom,
        sendDirectToRoomPeer,
        getLocalPeerId: () => localPeerId,
        getDirectConnections: () => Array.from(directConnections.keys()),
    };
})();

export default MPLib;