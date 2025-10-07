from playwright.sync_api import sync_playwright

def run_minimal_test(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    console_messages = []
    # Set up the listener *before* navigating.
    page.on("console", lambda msg: console_messages.append(msg.text))

    try:
        print("Navigating to a simple data: URI with a console.log...")
        # This URI is a self-contained HTML document with one script.
        page.goto("data:text/html,<script>console.log('test');</script>")

        # Give it a moment to ensure the log is processed.
        page.wait_for_timeout(1000)

        print("\n--- Captured Browser Console Output ---")
        if not console_messages:
            print("No console messages were captured.")
        else:
            full_console_log = "\n".join(console_messages)
            print(full_console_log)
        print("--------------------------------------")

        # Check for the specific test message.
        test_message_found = any('test' in msg for msg in console_messages)
        if test_message_found:
            print("\nSUCCESS: Minimal test message found. Playwright log capturing is working.")
        else:
            raise Exception("FAILURE: Minimal test message NOT found. Playwright log capturing is NOT working.")

        print("\nVerification script completed successfully.")

    except Exception as e:
        print(f"An error occurred during verification: {e}")
        raise
    finally:
        browser.close()

with sync_playwright() as playwright:
    run_minimal_test(playwright)