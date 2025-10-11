import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the local file first
        await page.goto(f"file://{os.path.abspath('index.html')}")

        # Now set the dummy API key in localStorage
        await page.evaluate('() => localStorage.setItem("apiKey", "DUMMY_KEY_FOR_TESTING")')

        # Reload the page for the key to be recognized by the app's initialization logic
        await page.reload()

        # Wait for a known-visible element to appear, confirming the page has loaded.
        await expect(page.get_by_role("heading", name="Strategic Direction")).to_be_visible(timeout=15000)

        # Take a screenshot to show the app is running
        await page.screenshot(path="jules-scratch/verification/verification.png")

        await browser.close()

if __name__ == "__main__":
    import os
    # The script needs to be run with the playwright python package
    # pip install playwright
    # playwright install
    asyncio.run(main())