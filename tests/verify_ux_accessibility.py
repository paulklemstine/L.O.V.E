
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

SERVER_PORT = 8001
SERVER_HOST = 'localhost'

def start_server():
    os.chdir('lovev1')
    server_address = (SERVER_HOST, SERVER_PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Server started on http://{SERVER_HOST}:{SERVER_PORT}")
    httpd.serve_forever()

def run_verification():
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Navigating to index.html...")
            page.goto(f"http://{SERVER_HOST}:{SERVER_PORT}/index.html")

            # Check if modal is visible
            if page.is_visible('#apiKeyModal'):
                print("Modal visible. Filling API key...")
                page.fill('#apiKeyInput', 'test-api-key')
                page.click('#saveApiKeyBtn')
                # Wait for modal to disappear
                page.wait_for_selector('#apiKeyModal', state='hidden')
                print("Modal dismissed.")

            # Wait for dashboard to load
            print("Waiting for panels...")
            page.wait_for_selector('.panel-header', state='attached', timeout=5000)

            headers = page.locator('.panel-header').all()
            print(f"Found {len(headers)} panel headers.")

            if len(headers) == 0:
                print("No headers found!")
                page.screenshot(path='debug_fail.png')
                sys.exit(1)

            failures = []

            for i, header in enumerate(headers):
                # Check attributes
                role = header.get_attribute('role')
                tabindex = header.get_attribute('tabindex')
                aria_expanded = header.get_attribute('aria-expanded')

                print(f"Header {i}: role={role}, tabindex={tabindex}, aria-expanded={aria_expanded}")

                if role != 'button':
                    failures.append(f"Header {i} missing role='button'")
                if tabindex != '0':
                    failures.append(f"Header {i} missing tabindex='0'")
                if aria_expanded is None:
                    failures.append(f"Header {i} missing aria-expanded")

            # Test Keyboard Interaction (Enter)
            first_header = headers[1] # Use second header (Strategic Direction) as first might be hidden
            # Check visibility
            if not first_header.is_visible():
                print("Header 1 is not visible. Using Header 2.")
                first_header = headers[2]

            first_panel = first_header.locator('xpath=..')

            # Initially verify collapsed state
            initial_class = first_panel.get_attribute('class')
            print(f"Initial panel class: {initial_class}")

            # Focus and press Enter
            print("Focusing and pressing Enter...")
            first_header.focus()
            page.keyboard.press('Enter')

            # Check if class changed
            new_class = first_panel.get_attribute('class')
            print(f"New panel class: {new_class}")

            if initial_class == new_class:
                 failures.append("Header did not toggle on Enter key press")

            # Test Keyboard Interaction (Space)
            # Toggle back
            print("Pressing Space...")
            page.keyboard.press('Space')
            final_class = first_panel.get_attribute('class')
            print(f"Final panel class: {final_class}")

            if final_class == new_class: # Should toggle back
                 failures.append("Header did not toggle on Space key press")

            if failures:
                print("\nVerification FAILED:")
                for f in failures:
                    print(f"- {f}")
                sys.exit(1)
            else:
                print("\nVerification PASSED!")
                sys.exit(0)

        except Exception as e:
            print(f"Error during verification: {e}")
            page.screenshot(path='debug_error.png')
            sys.exit(1)
        finally:
            browser.close()

if __name__ == '__main__':
    run_verification()
