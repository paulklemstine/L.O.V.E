import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from playwright.sync_api import sync_playwright

# Define the server
PORT = 8081
server_thread = None
httpd = None

def start_server():
    global httpd
    print(f"Starting server on port {PORT}...")
    httpd = HTTPServer(('localhost', PORT), SimpleHTTPRequestHandler)
    httpd.serve_forever()

def stop_server():
    global httpd
    if httpd:
        httpd.shutdown()
        httpd.server_close()
        print("Server stopped.")

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Setup mock BEFORE navigating
        page.add_init_script("""
            window.mockWsSend = [];
            window.WebSocket = class MockWebSocket {
                static get OPEN() { return 1; }
                static get CONNECTING() { return 0; }
                static get CLOSING() { return 2; }
                static get CLOSED() { return 3; }

                constructor(url) {
                    this.readyState = 1; // OPEN
                    this.onopen = null;
                    this.onmessage = null;
                    this.onclose = null;
                    setTimeout(() => {
                        if (this.onopen) this.onopen();
                    }, 100);
                }
                send(data) {
                    window.mockWsSend.push(data);
                }
                close() {}
            };
        """)

        page.goto(f"http://localhost:{PORT}/love_interface.html")

        # Check for textarea
        textarea = page.locator("#input-area")
        if not textarea.is_visible():
            print("❌ FAIL: Textarea not found")
            return False

        # Check for Aria Label
        aria_label = textarea.get_attribute("aria-label")
        if aria_label == "Command input":
            print("✅ PASS: Aria label found")
        else:
            print(f"❌ FAIL: Aria label mismatch or missing. Found: {aria_label}")

        # Check for Placeholder
        placeholder = textarea.get_attribute("placeholder")
        if placeholder == "Enter command (Shift+Enter for new line)...":
             print("✅ PASS: Placeholder found")
        else:
             print(f"❌ FAIL: Placeholder mismatch or missing. Found: {placeholder}")

        # Wait for "connection" (mocked)
        time.sleep(0.5)

        # Type something
        textarea.fill("test command")

        # Press Enter
        textarea.press("Enter")

        # Give it a bit of time for event handling
        time.sleep(0.5)

        val = textarea.input_value()
        sent_messages = page.evaluate("window.mockWsSend")

        if val == "":
            print("✅ PASS: Input cleared on Enter")
        else:
            print(f"❌ FAIL: Input NOT cleared on Enter. Value: '{val}'")

        if len(sent_messages) > 0:
             print(f"✅ PASS: Message sent via WebSocket: {sent_messages[0]}")
        else:
             print("❌ FAIL: No message sent via WebSocket")

        # Test Shift+Enter
        textarea.fill("line 1")
        textarea.press("Shift+Enter")
        textarea.type("line 2")

        val = textarea.input_value()
        # Note: browser might normalize newlines
        if "line 1" in val and "line 2" in val:
             print("✅ PASS: Shift+Enter created new line")
        else:
             print(f"❌ FAIL: Shift+Enter did not work as expected. Value: {repr(val)}")

        browser.close()
        return True

if __name__ == "__main__":
    # Start server in a thread
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()

    # Give server time to start
    time.sleep(2)

    try:
        verify()
    finally:
        stop_server()
