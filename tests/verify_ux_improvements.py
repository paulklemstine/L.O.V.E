import os
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

# Start a simple HTTP server to serve the HTML file
def start_server():
    server = HTTPServer(('localhost', 8081), SimpleHTTPRequestHandler)
    server.serve_forever()

def run_test():
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    # Give the server a moment to start
    time.sleep(1)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Mock WebSocket to capture sent messages
        page.add_init_script("""
            let mockWs = {
                send: (msg) => { window.lastMessage = msg; },
                close: () => {},
                readyState: 1 // OPEN
            };
            window.WebSocket = function(url) {
                setTimeout(() => {
                    if (this.onopen) this.onopen();
                }, 10);
                this.send = mockWs.send;
                this.close = mockWs.close;
                this.readyState = 1;
                return this;
            };
            // Restore static constants
            window.WebSocket.CONNECTING = 0;
            window.WebSocket.OPEN = 1;
            window.WebSocket.CLOSING = 2;
            window.WebSocket.CLOSED = 3;
        """)

        page.goto("http://localhost:8081/love_interface.html")

        print("Checking initial state...")

        # Check ARIA labels
        input_area = page.locator("#input-area")
        cognitive_stream = page.locator("#cognitive-stream")

        aria_label = input_area.get_attribute("aria-label")
        role = cognitive_stream.get_attribute("role")
        aria_live = cognitive_stream.get_attribute("aria-live")

        print(f"Input aria-label: {aria_label}")
        print(f"Stream role: {role}")
        print(f"Stream aria-live: {aria_live}")

        # Wait a bit for the mock connection to 'open'
        page.wait_for_timeout(100)

        # Test 1: Enter to send
        print("Testing Enter to send...")
        input_area.fill("Hello World")
        input_area.press("Enter")

        # Check if message was sent (we need to access window.lastMessage)
        last_message = page.evaluate("window.lastMessage")
        print(f"Message after Enter: {last_message}")

        # Reset
        page.evaluate("window.lastMessage = null")
        input_area.fill("")

        # Test 2: Shift+Enter for newline
        print("Testing Shift+Enter...")
        input_area.fill("Line 1")
        input_area.press("Shift+Enter")
        input_area.type("Line 2")

        val = input_area.input_value()
        print(f"Input value after Shift+Enter: {repr(val)}")

        last_message_after_shift = page.evaluate("window.lastMessage")
        print(f"Message after Shift+Enter: {last_message_after_shift}")

        browser.close()

if __name__ == "__main__":
    run_test()
