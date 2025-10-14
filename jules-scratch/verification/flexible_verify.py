import asyncio
import os
import re
from playwright.async_api import async_playwright, expect

async def main():
    """
    This script verifies the P2P lobby functionality with flexibility.
    It checks if the first instance becomes a host OR connects as a client.
    The second instance should always connect as a client.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        print("--- Verifying P2P Lobby Logic with Flexibility ---")

        # 1. Launch the FIRST instance
        context1 = await browser.new_context()
        page1 = await context1.new_page()

        await page1.add_init_script("localStorage.setItem('apiKey', 'dummy-key-for-testing');")

        index_path = f"file://{os.path.abspath('index.html')}"
        await page1.goto(index_path)

        # Assert that the first instance becomes EITHER the host OR connects successfully
        try:
            await expect(page1.locator("#p2p-status-text")).to_have_text("Lobby Host", timeout=15000)
            print("✅ Instance 1 became the Lobby Host.")
        except AssertionError:
            await expect(page1.locator("#p2p-status-text")).to_have_text(re.compile("Successfully connected to lobby"), timeout=15000)
            print("✅ Instance 1 connected as a Client (Host already exists).")

        await page1.screenshot(path="jules-scratch/verification/instance1_status.png")

        # Give the network a moment to settle
        await page1.wait_for_timeout(2000)

        # 2. Launch the SECOND instance
        context2 = await browser.new_context()
        page2 = await context2.new_page()

        await page2.add_init_script("localStorage.setItem('apiKey', 'dummy-key-for-testing');")
        await page2.goto(index_path)

        # The second instance should ALWAYS connect as a client
        await expect(page2.locator("#p2p-status-text")).to_have_text(re.compile("Successfully connected to lobby"), timeout=15000)
        await page2.screenshot(path="jules-scratch/verification/instance2_status.png")
        print("✅ Instance 2 connected as a Client.")

        # 3. Clean up
        await context1.close()
        await context2.close()
        await browser.close()

        print("\n🎉 Flexible verification successful! 🎉")

if __name__ == "__main__":
    asyncio.run(main())