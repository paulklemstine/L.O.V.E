import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Set localStorage before navigating to the page
        await page.add_init_script("""
            localStorage.setItem('jules-saved-pages', JSON.stringify([
                { id: 'test-page-12345', code: '<html></html>', cid: 'QmTestCID12345' }
            ]));
            localStorage.setItem('apiKey', 'dummy-key');
        """)

        # Log console messages from the page to help with debugging
        page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))

        # Navigate to the local server
        await page.goto("http://localhost:8000/index.html")

        # Let's check the localStorage directly after navigation
        saved_pages_from_browser = await page.evaluate("() => localStorage.getItem('jules-saved-pages')")
        print(f"LocalStorage content from browser: {saved_pages_from_browser}")

        # The assertion will now wait for the list item with the specific text to appear.
        # This is a more robust way to check for the rendered output.
        await expect(page.locator("#saved-pages-list li")).to_contain_text('QmTestCID12345', timeout=5000)

        # Ensure the panel is visible for the screenshot
        await page.evaluate("document.querySelector('#saved-pages-list').closest('.panel').classList.remove('collapsed')")

        # Take a screenshot
        await page.screenshot(path="jules-scratch/verification/verification.png")

        await browser.close()

if __name__ == "__main__":
    import os
    asyncio.run(main())