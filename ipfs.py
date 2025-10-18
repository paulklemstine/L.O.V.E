import logging
import random
from concurrent.futures import as_completed, ThreadPoolExecutor
import subprocess
import sys
import asyncio
import aiohttp
import time
import uuid


def _install_dependencies():
    """Installs the aioipfs library if not already installed."""
    try:
        import aioipfs
    except ImportError:
        print("aioipfs library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aioipfs"])

_install_dependencies()

import aioipfs
from rich.console import Console

from bbs import run_hypnotic_progress


async def get_ipfs_client(console):
    """Initializes and returns an IPFS client, handling potential errors."""
    try:
        # Attempt to connect to the default API address
        client = aioipfs.AsyncIPFS(maddr='/ip4/127.0.0.1/tcp/5002')
        await client.version() # A simple command to check if the daemon is responsive
        logging.info("Successfully connected to IPFS daemon.")
        return client
    except aiohttp.client_exceptions.ClientConnectorError:
        logging.error("IPFS daemon not running or API is not accessible.")
        if console:
            console.print("[bold red]IPFS Error:[/bold red] Could not connect to the IPFS daemon.")
            console.print("[yellow]Please ensure the IPFS daemon is running (`ipfs daemon`) and accessible.[/yellow]")
        else:
            print("ERROR: Could not connect to the IPFS daemon. Please ensure it's running.")
        return None
    except Exception as e:
        logging.critical(f"An unexpected error occurred while connecting to IPFS: {e}")
        if console:
            console.print(f"[bold red]An unexpected and critical error occurred with IPFS: {e}[/bold red]")
        else:
            print(f"CRITICAL IPFS ERROR: {e}")
        return None

async def pin_to_ipfs(content, console=None):
    """Adds and pins content (bytes) to IPFS, returning the IPFS hash (CID)."""
    client = await get_ipfs_client(console)
    if not client:
        return None

    try:
        result = await client.add_bytes(content)
        cid = result['Hash']
        logging.info(f"Content successfully pinned to IPFS with CID: {cid}")
        return cid
    except Exception as e:
        logging.error(f"Failed to pin content to IPFS: {e}")
        if console:
            console.print(f"[bold red]IPFS pinning failed:[/bold red] {e}")
        else:
            print(f"IPFS pinning failed: {e}")
        return None
    finally:
        if client:
            await client.close()

def pin_to_ipfs_sync(content, console: Console):
    """
    Synchronously pins content to the local IPFS node using requests and returns the CID.
    The content can be a filepath (str) or raw data (bytes).
    """
    if not ipfs_daemon_running_sync():
        return None

    try:
        if isinstance(content, str) and os.path.exists(content):
            with open(content, 'rb') as f:
                files = {'file': f}
                response = requests.post("http://127.0.0.1:5002/api/v0/add", files=files, params={'pin': 'true'})
        elif isinstance(content, bytes):
            files = {'file': content}
            response = requests.post("http://127.0.0.1:5002/api/v0/add", files=files, params={'pin': 'true'})
        else:
            console.print("[bold red]IPFS Sync Error: Invalid content type for pinning.[/bold red]")
            return None

        response.raise_for_status()
        result = response.json()
        cid = result.get('Hash')
        return cid
    except requests.exceptions.RequestException as e:
        # console.print(f"[bold red]An error occurred while pinning to IPFS (sync): {e}[/bold red]")
        return None
    except Exception as e:
        # console.print(f"[bold red]A general error occurred in pin_to_ipfs_sync: {e}[/bold red]")
        return None

def ipfs_daemon_running_sync():
    """Synchronously checks if the IPFS daemon is responsive."""
    try:
        response = requests.post("http://127.0.0.1:5002/api/v0/id")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


async def get_from_ipfs(cid, console=None):
    """Retrieves content from IPFS given a CID."""
    client = await get_ipfs_client(console)
    if not client:
        return None

    try:
        content = await client.cat(cid, timeout=30)
        logging.info(f"Successfully retrieved content from IPFS for CID: {cid}")
        return content
    except aioipfs.exceptions.Error as e:
        # This handles cases where the CID is not found or other IPFS errors
        logging.error(f"Failed to retrieve content from IPFS for CID {cid}: {e}")
        if console:
            console.print(f"[bold red]IPFS Retrieval Failed:[/bold red] Could not get content for CID [white]{cid}[/white].")
            console.print(f"[red]Error details: {e}[/red]")
        else:
            print(f"IPFS Retrieval Failed for CID {cid}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during IPFS retrieval: {e}")
        if console:
            console.print(f"[bold red]An unexpected error occurred during IPFS retrieval: {e}[/bold red]")
        else:
            print(f"An unexpected error occurred during IPFS retrieval: {e}")
        return None
    finally:
        if client:
            await client.close()


async def verify_ipfs_pin(cid, console):
    """Verifies a CID is available on public gateways."""
    gateways = [
        "https://ipfs.io/ipfs/",
        "https://gateway.pinata.cloud/ipfs/",
        "https://cloudflare-ipfs.com/ipfs/",
    ]
    random.shuffle(gateways)

    logging.info(f"Verifying CID {cid} on public gateways...")
    if console:
        console.print(f"Verifying CID [bold white]{cid}[/bold white] on public gateways...")

    async def _verify_task():
        """The actual verification logic for a single gateway."""
        async with aiohttp.ClientSession() as session:
            tasks = [session.head(f"{gateway}{cid}", timeout=20) for gateway in gateways]
            for future in asyncio.as_completed(tasks):
                try:
                    response = await future
                    if response.status >= 200 and response.status < 300:
                        logging.info(f"CID {cid} confirmed on gateway: {response.url.host}")
                        return True, str(response.url)
                except Exception as e:
                    logging.warning(f"Gateway failed to verify CID {cid}: {e}")
            return False, None

    try:
        if console:
            verified, gateway_url = await run_hypnotic_progress(
                f"Confirming network propagation for CID...",
                _verify_task()
            )
        else:
            print("Confirming network propagation...")
            verified, gateway_url = await _verify_task()

        if verified:
            if console:
                console.print(f"[bold green]Propagation confirmed on gateway:[/bold green] [underline]{gateway_url}[/underline]")
            else:
                print(f"Propagation confirmed on: {gateway_url}")
            return True
        else:
            if console:
                console.print("[bold yellow]Warning:[/bold yellow] Could not confirm CID on any public gateways. It may take more time to propagate.")
            else:
                print("Warning: Could not confirm CID on public gateways.")
            logging.warning(f"Failed to verify CID {cid} on all tested gateways.")
            return False
    except Exception as e:
        if console:
            console.print(f"[bold red]An unexpected error occurred during IPFS verification: {e}[/bold red]")
        else:
            print(f"An unexpected error occurred during IPFS verification: {e}")
        logging.error(f"Unexpected verification error for CID {cid}: {e}")
        return False

# --- Decentralized Storage with Encryption ---

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import json

class DecentralizedStorage:
    """
    Manages storing and retrieving encrypted data on IPFS.
    - Encrypts data using a public RSA key for confidentiality.
    - Decrypts data using the corresponding private RSA key.
    - Pins and retrieves data from an IPFS node.
    """
    def __init__(self, ipfs_client, public_key_path="creator_public.pem", private_key_path="creator_private.pem", console=None):
        self.client = ipfs_client
        self.console = console
        self.public_key = self._load_public_key(public_key_path)
        self.private_key = self._load_private_key(private_key_path)

    def _load_public_key(self, key_path):
        """Loads an RSA public key from a PEM file."""
        try:
            with open(key_path, "rb") as f:
                return RSA.import_key(f.read())
        except (IOError, ValueError) as e:
            if self.console:
                self.console.print(f"[bold red]Error loading public key '{key_path}': {e}[/bold red]")
            logging.error(f"Failed to load public key from {key_path}: {e}")
            return None

    def _load_private_key(self, key_path):
        """Loads an RSA private key from a PEM file."""
        try:
            with open(key_path, "rb") as f:
                return RSA.import_key(f.read())
        except (IOError, ValueError) as e:
            # Private key is optional for storing data, so this is a warning.
            if self.console:
                self.console.print(f"[yellow]Warning: Could not load private key '{key_path}'. Retrieval will not be possible. {e}[/yellow]")
            logging.warning(f"Could not load private key from {key_path}: {e}")
            return None

    def store_data(self, data: bytes) -> str | None:
        """
        Encrypts data with the public key and pins it to IPFS.
        Returns the CID of the stored, encrypted data.
        """
        if not self.public_key:
            if self.console:
                self.console.print("[bold red]Cannot store data: Public key is not loaded.[/bold red]")
            return None
        if not self.client:
            if self.console:
                self.console.print("[bold red]Cannot store data: IPFS client is not available.[/bold red]")
            return None

        try:
            # Encrypt the data
            cipher_rsa = PKCS1_OAEP.new(self.public_key)
            encrypted_data = cipher_rsa.encrypt(data)

            # Pin the encrypted data to IPFS
            cid = pin_to_ipfs(encrypted_data, self.console)
            return cid
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Failed to encrypt and store data: {e}[/bold red]")
            logging.error(f"Failed during data encryption and IPFS pinning: {e}")
            return None

    def retrieve_data(self, cid: str) -> bytes | None:
        """
        Retrieves encrypted data from IPFS and decrypts it with the private key.
        Returns the original, decrypted data.
        """
        if not self.private_key:
            if self.console:
                self.console.print("[bold red]Cannot retrieve data: Private key is not loaded.[/bold red]")
            return None
        if not self.client:
            if self.console:
                self.console.print("[bold red]Cannot retrieve data: IPFS client is not available.[/bold red]")
            return None

        try:
            # Get the encrypted data from IPFS
            encrypted_data = get_from_ipfs(cid, self.console)
            if not encrypted_data:
                return None # Error message is handled by get_from_ipfs

            # Decrypt the data
            cipher_rsa = PKCS1_OAEP.new(self.private_key)
            decrypted_data = cipher_rsa.decrypt(encrypted_data)
            return decrypted_data
        except ValueError as e:
            # This commonly occurs if the wrong key is used for decryption
            if self.console:
                self.console.print(f"[bold red]Decryption failed for CID {cid}. The data may be corrupted or the wrong key is being used.[/bold red]")
                self.console.print(f"[red]Error details: {e}[/red]")
            logging.error(f"Decryption failed for CID {cid}: {e}")
            return None
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Failed to retrieve and decrypt data for CID {cid}: {e}[/bold red]")
            logging.error(f"Failed during data retrieval and decryption for CID {cid}: {e}")
            return None

class DataManifest:
    """
    Manages a manifest of data stored in decentralized storage.
    The manifest is a JSON file that tracks metadata for each piece of data,
    including its description, CID, and timestamp. The manifest itself is
    stored on IPFS, providing a single, updatable entry point to all
    of the agent's decentralized intelligence.
    """
    def __init__(self, storage: DecentralizedStorage, console=None):
        self.storage = storage
        self.console = console
        self.manifest_cid = None
        self.manifest_data = {"version": "1.0", "entries": {}}

    def load_manifest(self, cid: str):
        """Loads an existing manifest from a given CID."""
        if self.console:
            self.console.print(f"Attempting to load manifest from CID: {cid}")

        decrypted_data = self.storage.retrieve_data(cid)
        if decrypted_data:
            try:
                self.manifest_data = json.loads(decrypted_data)
                self.manifest_cid = cid
                if self.console:
                    self.console.print(f"[green]Successfully loaded and decrypted manifest.[/green]")
                return True
            except json.JSONDecodeError as e:
                if self.console:
                    self.console.print(f"[bold red]Failed to parse manifest JSON from CID {cid}: {e}[/bold red]")
                return False
        return False

    def add_entry(self, description: str, cid: str, data_type: str = "intelligence_report"):
        """Adds a new entry to the manifest and saves the updated manifest."""
        entry_id = str(uuid.uuid4())
        self.manifest_data["entries"][entry_id] = {
            "description": description,
            "cid": cid,
            "data_type": data_type,
            "timestamp": time.time()
        }
        return self.save_manifest()

    def save_manifest(self) -> str | None:
        """Saves the current manifest to IPFS and returns the new manifest CID."""
        try:
            manifest_bytes = json.dumps(self.manifest_data, indent=4).encode('utf-8')
            new_cid = self.storage.store_data(manifest_bytes)
            if new_cid:
                self.manifest_cid = new_cid
                if self.console:
                    self.console.print(f"Manifest saved. New CID: [bold white]{new_cid}[/bold white]")
            return new_cid
        except Exception as e:
            if self.console:
                self.console.print(f"[bold red]Failed to save manifest: {e}[/bold red]")
            return None

    def get_entry(self, entry_id: str):
        """Retrieves a specific entry from the manifest."""
        return self.manifest_data["entries"].get(entry_id)

    def get_all_entries(self):
        """Returns all entries in the manifest."""
        return self.manifest_data["entries"]