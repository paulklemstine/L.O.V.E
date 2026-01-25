const MPLib = (() => {
    const LOVE_LOBBY_TOPIC = 'love-lobby';
    let config = {};
    let libp2p = null;
    let localPeerId = null;

    const defaultConfig = {
        onStatusUpdate: (msg, type) => console.log(`[MPLib] ${msg}`),
        onError: (type, err) => console.error(`[MPLib] Error (${type}):`, err),
        onConnectionChange: (count) => {},
        onRoomDataReceived: (peerId, data) => {},
    };

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

    async function initialize(options = {}) {
        config = { ...defaultConfig, ...options };
        logMessage("Initializing MPLib with libp2p...", 'info');

        libp2p = await Libp2p.create({
            addresses: {
                listen: [
                    '/p2p-circuit',
                    '/webrtc'
                ]
            },
            transports: [
                Libp2p.websockets(),
                Libp2p.webrtc(),
                Libp2p.circuitRelayTransport()
            ],
            connectionEncryption: [Libp2p.noise()],
            streamMuxers: [Libp2p.yamux()],
            connectionGater: {
                denyDialMultiaddr: () => {
                    return false
                }
            },
            services: {
                identify: Libp2p.identify(),
                pubsub: Libp2p.floodsub()
            }
        });

        localPeerId = libp2p.peerId.toString();
        logMessage(`Peer ID: ${localPeerId}`, 'status');

        libp2p.addEventListener('connection:open', (evt) => {
            logMessage(`Connected to ${evt.detail.remotePeer.toString()}`, 'info');
            config.onConnectionChange(libp2p.getPeers().length);
        });

        libp2p.addEventListener('connection:close', (evt) => {
            logMessage(`Disconnected from ${evt.detail.remotePeer.toString()}`, 'info');
            config.onConnectionChange(libp2p.getPeers().length);
        });

        libp2p.services.pubsub.subscribe(LOVE_LOBBY_TOPIC);
        libp2p.services.pubsub.addEventListener('message', (evt) => {
            const from = evt.detail.from.toString();
            if (from === localPeerId) {
                return;
            }
            try {
                const data = JSON.parse(new TextDecoder().decode(evt.detail.data));
                const decrypted = JSON.parse(decrypt(data.payload));
                config.onRoomDataReceived(from, decrypted);
            } catch (e) {
                logMessage(`Failed to decrypt or parse message from ${from}`, 'error');
            }
        });
    }

    async function connectToRelay(multiaddr) {
        try {
            await libp2p.dial(multiaddr);
            logMessage(`Connected to relay: ${multiaddr}`, 'status');
        } catch (err) {
            config.onError('relay-connection', err);
        }
    }

    function broadcastToRoom(data) {
        const encrypted = { type: 'encrypted-message', payload: encrypt(JSON.stringify(data)) };
        libp2p.services.pubsub.publish(LOVE_LOBBY_TOPIC, new TextEncoder().encode(JSON.stringify(encrypted)));
    }

    return {
        initialize,
        connectToRelay,
        broadcastToRoom,
        getLocalPeerId: () => localPeerId,
        getPeers: () => libp2p.getPeers().map(p => p.toString()),
    };
})();

export default MPLib;
