import threading
import http.server
import socketserver
import os
import sys
from playwright.sync_api import sync_playwright, expect

# 1. Start a temporary HTTP server
PORT = 0
Handler = http.server.SimpleHTTPRequestHandler
httpd = None
server_thread = None

def start_server():
    global httpd, PORT
    # Use port 0 to let OS assign a free port
    with socketserver.TCPServer(("", 0), Handler) as server:
        PORT = server.server_address[1]
        httpd = server
        print(f"Serving at port {PORT}")
        server.serve_forever()

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        url = f"http://localhost:{PORT}/core/web/static/index.html"
        print(f"Navigating to {url}")

        # Mock API calls to avoid 404s/Errors cluttering console
        page.route("**/api/**", lambda route: route.fulfill(status=200, body='{}', content_type='application/json'))

        page.goto(url)

        # Check initial state
        header = page.locator("#logs-header")
        expect(header).to_be_visible()
        expect(header).to_have_attribute("role", "button")
        expect(header).to_have_attribute("tabindex", "0")
        expect(header).to_have_attribute("aria-expanded", "false")

        print("✅ Initial state verified")

        # Test Interaction (Click)
        header.click()
        expect(header).to_have_attribute("aria-expanded", "true")
        print("✅ Click interaction verified (expanded=true)")

        # Test Interaction (Keyboard Enter)
        header.focus()
        page.keyboard.press("Enter")
        expect(header).to_have_attribute("aria-expanded", "false")
        print("✅ Keyboard Enter interaction verified (expanded=false)")

        # Test Interaction (Keyboard Space)
        header.focus()
        page.keyboard.press("Space")
        expect(header).to_have_attribute("aria-expanded", "true")
        print("✅ Keyboard Space interaction verified (expanded=true)")

        # Take screenshot
        os.makedirs("verification", exist_ok=True)
        screenshot_path = "verification/logs_header.png"
        page.screenshot(path=screenshot_path)
        print(f"📸 Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    # Start server in background thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    import time
    time.sleep(1) # Wait for server to start

    try:
        verify_frontend()
    finally:
        if httpd:
            httpd.shutdown()
