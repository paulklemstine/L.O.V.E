# AGENTS.md - WebVM & Frontend interface

## Purpose
The "Body" and "Interface" of L.O.V.E. accessible to users via the browser. This environment runs a Linux emulator in the browser.

## Constraints
> [!IMPORTANT]
> **Browser Sandbox**: This code runs inside a browser tab.
- **Networking**: No raw sockets available. You must use WebSocket proxies for TCP/UDP.
- **Filesystem**: The filesystem is ephemeral (in-memory) unless explicitly synced to IndexedDB.
- **Resources**: CPU and RAM are limited by the user's browser tab.

## Architecture
- `index.html`: The entry point.
- `mplib.js`: Message passing library for communicating with the Core backend.
