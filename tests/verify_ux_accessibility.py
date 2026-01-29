import threading
import http.server
import socketserver
import os
import sys
from playwright.sync_api import sync_playwright, expect

# Set the working directory to the repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(REPO_ROOT)

PORT = 0  # Dynamic port

def run_server():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        global SERVER_PORT
        SERVER_PORT = httpd.server_address[1]
        print(f"Serving at port {SERVER_PORT}")
        httpd.serve_forever()

def verify_ux():
    # Start server in a thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to start
    import time
    time.sleep(2)

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Subscribe to console events
            page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))

            url = f"http://localhost:{SERVER_PORT}/lovev1/index.html"
            print(f"Navigating to {url}")

            # Navigate
            page.goto(url)

            # Check if API key modal is present
            if page.locator("#apiKeyModal").is_visible():
                print("API Key Modal found. Bypassing...")
                # Inject API key into localStorage
                page.evaluate("localStorage.setItem('apiKey', 'dummy-key')")
                page.reload()

            # Wait for dashboard
            expect(page.locator("#dashboard")).to_be_visible()
            print("Dashboard visible.")

            # Find panel headers
            headers = page.locator(".panel-header")
            count = headers.count()
            print(f"Found {count} panel headers.")

            # Find first visible header
            first_header = None
            for i in range(count):
                header = headers.nth(i)
                if header.is_visible():
                    first_header = header
                    print(f"Found visible header at index {i}")
                    break

            if not first_header:
                raise Exception("No visible panel headers found")

            # Check for accessibility attributes
            print("Checking accessibility attributes...")

            role = first_header.get_attribute("role")
            tabindex = first_header.get_attribute("tabindex")
            aria_expanded = first_header.get_attribute("aria-expanded")

            print(f"Role: {role}")
            print(f"Tabindex: {tabindex}")
            print(f"Aria-Expanded: {aria_expanded}")

            assert role == "button", f"Expected role='button', but got '{role}'"
            assert tabindex == "0", f"Expected tabindex='0', but got '{tabindex}'"
            assert aria_expanded is not None, "Expected aria-expanded to be present"

            print("Attributes verified successfully.")

            # Test Keyboard Interaction
            print("Testing keyboard interaction...")
            # Focus on the first header
            first_header.focus()

            # Check active element
            active_tag = page.evaluate("document.activeElement.tagName")
            active_class = page.evaluate("document.activeElement.className")
            print(f"Active Element: {active_tag}.{active_class}")

            # Initial state should be expanded (not collapsed) or collapsed.
            # Let's assume it starts expanded (aria-expanded="true") based on logic.
            # But wait, some panels might start collapsed?
            # The JS: const isCollapsed = panel.classList.contains('collapsed');
            # If panel has .collapsed, expanded is false.

            initial_expanded = first_header.get_attribute("aria-expanded")
            print(f"Initial state: {initial_expanded}")

            # Press Enter
            page.keyboard.press("Enter")

            # Wait for state change
            page.wait_for_timeout(500) # Wait for JS execution

            new_expanded = first_header.get_attribute("aria-expanded")
            print(f"Post-Enter state: {new_expanded}")

            assert initial_expanded != new_expanded, "State did not toggle on Enter key"

            print("Keyboard interaction verified.")

            # Take a screenshot
            page.screenshot(path="tests/ux_verification_after.png")

        except Exception as e:
            print(f"Error: {e}")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_ux()
