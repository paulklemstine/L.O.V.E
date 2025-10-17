import os
import json
import shutil
import tarfile
import tempfile
import time
import requests
from threading import Thread
from rich.console import Console

# --- Constants ---
STATE_FILE = "love_state.json"
# Public gateways to verify IPFS pins
IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/{}",
    "https://dweb.link/ipfs/{}",
    "https://gateway.pinata.cloud/ipfs/{}",
]

# --- IPFS Helper ---
def _get_ipfs_client(console: Console):
    """Attempts to connect to the IPFS daemon."""
    try:
        import ipfshttpclient
        return ipfshttpclient.connect(timeout=10)
    except (ImportError, Exception):
        # This is not an error, but a silent failure for environments without IPFS.
        return None

def _pin_to_ipfs(client, data_bytes: bytes, console: Console):
    """Pins raw bytes to IPFS and returns the CID string."""
    if not client:
        return None
    try:
        # client.add_bytes returns a dict, e.g., {'Hash': 'Qm...', 'Name': '...'}
        result_hash = client.add_bytes(data_bytes)
        console.print(f"[cyan]IPFS:[/cyan] Pinned data, received CID: {result_hash}")
        return str(result_hash)
    except Exception as e:
        console.print(f"[bold red]IPFS Error: Failed to pin data. {e}[/bold red]")
        return None

def _verify_pin_on_gateways(cid: str, console: Console):
    """Asynchronously checks for a CID on public gateways."""
    def verify():
        time.sleep(5) # Give some time for the pin to propagate
        for gateway_url in IPFS_GATEWAYS:
            url = gateway_url.format(cid)
            try:
                response = requests.head(url, timeout=20)
                if response.status_code == 200:
                    console.print(f"[green]IPFS Verification:[/green] CID {cid[:15]}... is available on {gateway_url.split('/')[2]}")
                    return # Exit after first successful verification
            except requests.RequestException:
                # Silently ignore failures, as gateways can be unreliable
                pass
        console.print(f"[yellow]IPFS Verification:[/yellow] Could not verify CID {cid[:15]}... on public gateways after checks.")

    # Run the verification in a background thread to avoid blocking
    Thread(target=verify, daemon=True).start()


# --- Main State Management Function ---
def save_all_state(love_state: dict, console: Console):
    """
    Orchestrates the saving of all critical application data.
    1. Backs up 'interesting files' from the knowledge base to IPFS.
    2. Saves the final, updated state to the local love_state.json file.
    3. Pins the state file itself to IPFS.
    4. Creates a manifest of all CIDs from this operation.
    """
    ipfs_client = _get_ipfs_client(console)

    try:
        # --- Prepare Manifest ---
        ipfs_manifest = {
            "timestamp": time.time(),
            "state_cid": None,
            "interesting_files_cids": {}
        }

        # 1. Pin all "interesting files" from the knowledge base
        fs_intel = love_state.get("knowledge_base", {}).get("file_system_intel", {})
        interesting_files = fs_intel.get("interesting_files", [])

        if interesting_files:
            console.print(f"[cyan]Backing up {len(interesting_files)} interesting files to IPFS...[/cyan]")
        for fpath in interesting_files:
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'rb') as f:
                        file_bytes = f.read()
                    cid = _pin_to_ipfs(ipfs_client, file_bytes, console)
                    if cid:
                        ipfs_manifest["interesting_files_cids"][fpath] = cid
                        _verify_pin_on_gateways(cid, console) # Asynchronously verify
                except IOError as e:
                    console.print(f"[yellow]Could not read interesting file '{fpath}': {e}[/yellow]")

        # 2. Update the main state with the new manifest
        love_state.setdefault("backup_manifests", []).append(ipfs_manifest)
        if len(love_state["backup_manifests"]) > 10: # Keep only the last 10 manifests
            love_state["backup_manifests"] = love_state["backup_manifests"][-10:]

        # 3. Save the final state file (locally and to IPFS)
        try:
            state_bytes_with_manifest = json.dumps(love_state, indent=4).encode('utf-8')

            # Pin the state itself to IPFS
            state_cid = _pin_to_ipfs(ipfs_client, state_bytes_with_manifest, console)
            if state_cid:
                ipfs_manifest["state_cid"] = state_cid
                love_state["state_cid"] = state_cid # Update the top-level CID
                _verify_pin_on_gateways(state_cid, console) # Asynchronously verify
                # Re-encode with the final state CID for local saving
                state_bytes_with_manifest = json.dumps(love_state, indent=4).encode('utf-8')

            # Write the final version to the local file
            with open(STATE_FILE, 'wb') as f:
                f.write(state_bytes_with_manifest)

            console.print(f"[bold green]Successfully saved state to '{STATE_FILE}'[/bold green]")
            if state_cid:
                console.print(f"[bold green]Final state CID: {state_cid}[/bold green]")

        except Exception as e:
            console.print(f"[bold red]CRITICAL: Failed to write final state to '{STATE_FILE}': {e}[/bold red]")

    finally:
        if ipfs_client:
            try:
                ipfs_client.close()
            except Exception:
                pass # Ignore errors on close

    return love_state