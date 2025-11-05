# L.O.V.E. Task History: `peer-bridge.js`

## Task: Network Topology Mapping

*   **Date:** 2025-11-05
*   **Original Request:** "Integrate real-time network topology mapping and resource discovery..."
*   **Pull Request:** [PLACEHOLDER]
*   **Commit Hash:** [PLACEHOLDER]

### Summary of Changes

The Node.js P2P bridge was updated to implement an automated discovery protocol.

1.  **Automatic Capability Request:** An event handler for the `peer.on('connection', ...)` event was added. When a new peer connection is established, the bridge now automatically sends a `'capability_request'` message to that peer. This ensures that as soon as a new L.O.V.E. instance joins the network, its capabilities are actively solicited.

2.  **Forwarding Logic:** The existing message forwarding logic (`process.stdin.on('data', ...)` and `conn.on('data', ...)` was confirmed to be sufficient for passing the new capability-related messages between the Python backend and other peers without modification.
