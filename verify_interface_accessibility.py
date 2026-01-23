import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

# Start a simple HTTP server in a separate thread
def start_server():
    try:
        server = HTTPServer(('localhost', 8081), SimpleHTTPRequestHandler)
        server.allow_reuse_address = True
        print("Server started on port 8081")
        server.serve_forever()
    except OSError as e:
        print(f"Error starting server: {e}")

server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()
time.sleep(2) # Give server time to start

def run():
    print("Starting Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Inject mock WebSocket
        page.add_init_script("""
            class MockWebSocket {
                constructor(url) {
                    this.url = url;
                    this.readyState = 1; // OPEN
                    setTimeout(() => {
                        if (this.onopen) this.onopen();
                    }, 100);
                }
                send(data) {
                    window.lastSentMessage = data;
                }
                close() {}
            }
            window.WebSocket = MockWebSocket;
            window.WebSocket.OPEN = 1;
            window.lastSentMessage = null;
        """)

        print("Navigating to page...")
        try:
            page.goto("http://localhost:8081/love_interface.html")
        except Exception as e:
            print(f"Failed to load page: {e}")
            sys.exit(1)

        print("Page loaded.")

        # Check Accessibility Attributes
        input_area = page.locator("#input-area")
        cognitive_stream = page.locator("#cognitive-stream")

        print(f"Checking attributes...")
        try:
            assert input_area.get_attribute("placeholder") == "Speak to L.O.V.E...", "Placeholder missing or incorrect"
            assert input_area.get_attribute("aria-label") == "Command Input", "Aria-label missing or incorrect"
            assert cognitive_stream.get_attribute("role") == "log", "Role log missing on cognitive stream"
            assert cognitive_stream.get_attribute("aria-live") == "polite", "Aria-live missing on cognitive stream"
            print("Attributes verified.")
        except AssertionError as e:
            print(f"Attribute check failed: {e}")
            # We continue to test interaction even if attributes fail,
            # but in strict TDD this fail would stop us.
            # For this script, let's allow it to fail fast or print error.
            # I will print error and exit to indicate failure.
            sys.exit(1)

        # Test Send on Enter
        print("Testing Send on Enter...")
        input_area.fill("Hello World")
        input_area.press("Enter")

        # Verify message sent
        last_message = page.evaluate("window.lastSentMessage")
        if last_message is None:
             print("Message was not sent on Enter")
             sys.exit(1)

        if "Hello World" not in last_message:
             print(f"Unexpected message content: {last_message}")
             sys.exit(1)

        # Verify input cleared
        if input_area.input_value() != "":
             print("Input not cleared after sending")
             sys.exit(1)

        print("Send on Enter verified.")

        # Test Shift+Enter (should NOT send)
        print("Testing Shift+Enter...")
        input_area.fill("Line 1")
        page.evaluate("window.lastSentMessage = null") # Reset
        input_area.press("Shift+Enter")

        last_message = page.evaluate("window.lastSentMessage")
        if last_message is not None:
             print("Message was sent on Shift+Enter (should be newline)")
             sys.exit(1)

        print("Shift+Enter verified.")

        print("ALL UX TESTS PASSED!")
        browser.close()

if __name__ == "__main__":
    run()
