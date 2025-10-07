from playwright.sync_api import sync_playwright, expect
import time
import re

def run_verification(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    console_messages = []
    # Capture console logs and store them
    page.on("console", lambda msg: console_messages.append(msg.text))

    try:
        # Navigate to the local server
        page.goto("http://localhost:8000/index.html")

        # Set a dummy API key in localStorage to bypass the modal
        page.evaluate("() => localStorage.setItem('apiKey', 'dummy-key')")

        # Reload the page to ensure the key is recognized
        page.reload()

        # Wait for the main container to be visible
        expect(page.locator("#container")).to_be_visible(timeout=10000)

        # --- Wait for Login Prompt ---
        print("Waiting for WebVM to boot and show login prompt...")
        max_wait_time = 120  # seconds
        start_time = time.time()
        login_prompt_found = False
        serial_output = ""

        while time.time() - start_time < max_wait_time:
            # Re-join the console messages each time to get the latest output
            full_console_log = "".join(console_messages)

            # Extract serial output using regex
            serial_output_chars = re.findall(r"WebVM Serial: (.*)", full_console_log)
            serial_output = "".join(serial_output_chars).replace('\\n', '\n').replace('\\r', '')

            if "login:" in serial_output:
                login_prompt_found = True
                print("SUCCESS: Login prompt found!")
                break
            time.sleep(2) # Poll every 2 seconds

        if not login_prompt_found:
            print("\n--- Captured WebVM Serial Output ---")
            print(serial_output)
            print("------------------------------------")
            raise Exception("FAILURE: Timed out waiting for login prompt.")

        # Find the necessary elements
        terminal_input = page.locator("#terminal-input")
        run_button = page.locator("#run-command-btn")
        debug_panel = page.locator("#debug-panel")

        # Hide the debug panel to prevent it from intercepting the click
        debug_panel.evaluate("element => element.style.display = 'none'")

        # Now that the VM is ready, enter a command
        command_to_run = "echo 'Hello from the terminal!'"
        print(f"Running command: {command_to_run}")
        terminal_input.fill(command_to_run)
        run_button.click()

        # Make the debug panel visible again
        debug_panel.evaluate("element => element.style.display = 'flex'")

        # Wait for the command to execute and output to be rendered
        print("Waiting for command output (10s)...")
        page.wait_for_timeout(10000)

        # Take a screenshot for visual verification
        screenshot_path = "jules-scratch/verification/terminal_verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        print("\nVerification script completed successfully. Please inspect the screenshot.")

    except Exception as e:
        print(f"An error occurred during verification: {e}")
        raise
    finally:
        browser.close()

with sync_playwright() as playwright:
    run_verification(playwright)