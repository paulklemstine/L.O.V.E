import logging
import random
from concurrent.futures import as_completed, ThreadPoolExecutor
import subprocess
import sys
import asyncio
import aiohttp


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
        client = aioipfs.AsyncIPFS(maddr='/ip4/127.0.0.1/tcp/5001')
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