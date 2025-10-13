const MPLib = (() => {
    // --- Overall State ---
    const API_KEY = 'peerjs'; // Public PeerJS API key
    const BORG_LOBBY_ID = 'borg-lobby';
    let config = {};

    // --- Peer Connection State ---
    let peer = null;
    let localPeerId = null;
    let evolvePyPeerConnection = null;

    // --- Callbacks & Config ---
    const defaultConfig = {
        debugLevel: 0,
        onStatusUpdate: (msg, type) => console.log(`[MPLib] ${msg}`),
        onError: (type, err) => console.error(`[MPLib] Error (${type}):`, err),
        onEvolvePeerConnected: (id) => {},
        onPinResponse: (data) => {},
    };

    function logMessage(message, type = 'info') {
        config.onStatusUpdate(message, type);
    }

    // --- Initialization ---
    function initialize(options = {}) {
        config = { ...defaultConfig, ...options };
        logMessage("Initializing MPLib and connecting to the borg-lobby...", 'info');

        peer = new Peer({ debug: config.debugLevel, key: API_KEY });

        peer.on('open', (id) => {
            localPeerId = id;
            logMessage(`PeerJS connection open with ID: ${id}. Connecting to lobby...`, 'info');
            connectToLobby();
        });

        peer.on('error', (err) => {
            config.onError('peer-error', err);
        });
    }

    function connectToLobby() {
        if (evolvePyPeerConnection && evolvePyPeerConnection.open) {
            logMessage('Already connected to lobby.', 'info');
            return;
        }

        logMessage(`Attempting to connect to lobby peer: ${BORG_LOBBY_ID}`, 'info');
        const conn = peer.connect(BORG_LOBBY_ID, { reliable: true });

        conn.on('open', () => {
            logMessage(`Successfully connected to lobby: ${conn.peer}`, 'success');
            evolvePyPeerConnection = conn;
            config.onEvolvePeerConnected(conn.peer);
            setupEvolvePeerConnection(conn);
        });

        conn.on('error', (err) => {
            logMessage(`Failed to connect to lobby: ${err.message}`, 'error');
        });
    }

    function setupEvolvePeerConnection(conn) {
        conn.on('data', (data) => {
            if (data.type === 'pin-response') {
                config.onPinResponse(data);
            } else if (data.type === 'capability-announcement' && data.payload.capability === 'evolve.py') {
                logMessage('Lobby confirmed evolve.py capability.', 'success');
            }
        });

        conn.on('close', () => {
            logMessage(`Connection to lobby closed.`, 'warn');
            evolvePyPeerConnection = null;
            // Optional: implement retry logic here
        });
    }

    function sendCodeToEvolvePeer(sourceCode) {
        if (evolvePyPeerConnection && evolvePyPeerConnection.open) {
            logMessage('Sending source code to evolve.py peer for IPFS pinning...', 'info');
            evolvePyPeerConnection.send({ type: 'pin-request', payload: sourceCode });
        } else {
            logMessage('Cannot send source code: Not connected to an evolve.py peer.', 'error');
            config.onError('send-code', 'Not connected to an evolve.py peer.');
        }
    }

    return {
        initialize,
        sendCodeToEvolvePeer,
        getLocalPeerId: () => localPeerId,
    };
})();

export default MPLib;