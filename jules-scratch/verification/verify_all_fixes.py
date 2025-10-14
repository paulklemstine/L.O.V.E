import asyncio
import os
import re
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # --- Test Case 1: creator.html becomes host, index.html connects as client ---
        print("--- Test Case 1: creator.html (host), index.html (client) ---")

        # 1. Launch creator.html as the host
        creator_context_host = await browser.new_context()
        creator_page_host = await creator_context_host.new_page()
        creator_path = f"file://{os.path.abspath('creator.html')}"
        await creator_page_host.goto(creator_path)

        # Wait for the host to be fully online by checking its log output
        await expect(creator_page_host.locator("#log")).to_contain_text("Lobby is online. Waiting for connections...", timeout=15000)
        await creator_page_host.screenshot(path="jules-scratch/verification/creator_as_host.png")
        print("✅ creator.html became host and is online.")

        # 2. Launch index.html as the client
        index_context_client = await browser.new_context()
        index_page_client = await index_context_client.new_page()
        index_path = f"file://{os.path.abspath('index.html')}"
        await index_page_client.add_init_script("localStorage.setItem('apiKey', 'dummy-key-for-testing');")
        await index_page_client.goto(index_path)

        # It should connect as a client since creator.html is already the host.
        await expect(index_page_client.locator("#p2p-status-text")).to_have_text(re.compile("Successfully connected to lobby"), timeout=15000)
        await index_page_client.screenshot(path="jules-scratch/verification/index_as_client.png")
        print("✅ index.html connected as client.")

        await creator_context_host.close()
        await index_context_client.close()

        # --- Test Case 2: index.html becomes host, creator.html connects as client ---
        print("\n--- Test Case 2: index.html (host), creator.html (client) ---")

        # 1. Launch index.html as the host
        index_context_host = await browser.new_context()
        index_page_host = await index_context_host.new_page()
        await index_page_host.add_init_script("localStorage.setItem('apiKey', 'dummy-key-for-testing');")
        await index_page_host.goto(index_path)

        # Wait for the host to be fully online by checking its status text
        await expect(index_page_host.locator("#p2p-status-text")).to_have_text("Lobby Host", timeout=15000)
        await index_page_host.screenshot(path="jules-scratch/verification/index_as_host.png")
        print("✅ index.html became host and is online.")

        # 2. Launch creator.html as the client
        creator_context_client = await browser.new_context()
        creator_page_client = await creator_context_client.new_page()
        await creator_page_client.goto(creator_path)
        await expect(creator_page_client.locator("#status")).to_have_text("Connected to lobby", timeout=15000)
        await creator_page_client.screenshot(path="jules-scratch/verification/creator_as_client_2.png")
        print("✅ creator.html connected as client.")

        await index_context_host.close()
        await creator_context_client.close()

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())