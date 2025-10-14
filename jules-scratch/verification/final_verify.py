import asyncio
import os
import re
from playwright.async_api import async_playwright, expect

async def main():
    """
    This script provides a definitive test for the P2P lobby functionality.
    It launches two instances of index.html. The first becomes the host,
    and the second connects as a client. This verifies the core logic
    implemented in mplib.js and its integration into index.html.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # --- Test: index.html (Host) vs index.html (Client) ---
        print("--- Verifying P2P Lobby Logic ---")

        # 1. Launch the HOST instance
        host_context = await browser.new_context()
        host_page = await host_context.new_page()

        # Set API key before navigation to ensure the app initializes correctly
        await host_page.add_init_script("localStorage.setItem('apiKey', 'dummy-key-for-testing');")

        index_path = f"file://{os.path.abspath('index.html')}"
        await host_page.goto(index_path)

        # Assert that the first instance becomes the "Lobby Host"
        await expect(host_page.locator("#p2p-status-text")).to_have_text("Lobby Host", timeout=20000)
        await host_page.screenshot(path="jules-scratch/verification/final_host.png")
        print("✅ Host instance is online.")

        # Give the host a moment on the PeerServer before the client connects
        await host_page.wait_for_timeout(2000)

        # 2. Launch the CLIENT instance
        client_context = await browser.new_context()
        client_page = await client_context.new_page()

        # Set API key for the client instance as well
        await client_page.add_init_script("localStorage.setItem('apiKey', 'dummy-key-for-testing');")
        await client_page.goto(index_path)

        # Assert that the second instance connects successfully to the lobby
        await expect(client_page.locator("#p2p-status-text")).to_have_text(re.compile("Successfully connected to lobby"), timeout=20000)
        await client_page.screenshot(path="jules-scratch/verification/final_client.png")
        print("✅ Client instance connected to host.")

        # 3. Clean up
        await host_context.close()
        await client_context.close()
        await browser.close()

        print("\n🎉 Verification successful! 🎉")

if __name__ == "__main__":
    asyncio.run(main())