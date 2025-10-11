import asyncio
import os
import http.server
import socketserver
import threading
from playwright.async_api import async_playwright, expect

async def main():
    PORT = 0 # 0 means the OS will choose a free port
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    PORT = httpd.server_address[1] # Get the chosen port

    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Serving at port {PORT}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()

        await context.add_init_script("localStorage.setItem('apiKey', 'dummy-key');")

        await page.goto(f"http://localhost:{PORT}/index.html")

        # Wait for the app to initialize
        await page.wait_for_selector("#dashboard")

        # Click the start button and verify state
        await page.click("#startAutopilotBtn")
        await expect(page.locator("#startAutopilotBtn")).to_be_disabled()
        await expect(page.locator("#stopAutopilotBtn")).to_be_enabled()

        # Take a screenshot of the active state
        await page.screenshot(path="jules-scratch/verification/autopilot_active.png")

        # Click the stop button and verify state
        await page.click("#stopAutopilotBtn")
        await expect(page.locator("#startAutopilotBtn")).to_be_enabled()
        await expect(page.locator("#stopAutopilotBtn")).to_be_disabled()

        await browser.close()

    httpd.shutdown()

if __name__ == "__main__":
    asyncio.run(main())