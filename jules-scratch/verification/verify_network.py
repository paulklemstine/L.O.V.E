from playwright.sync_api import sync_playwright
import time

def run_network_test(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # --- Event Listeners ---
    console_messages = []
    page.on("console", lambda msg: console_messages.append(msg.text))

    print("--- Network Log ---")
    page.on("request", lambda request: print(f">> {request.method} {request.url}"))
    page.on("response", lambda response: print(f"<< {response.status} {response.url}\n   Headers: {response.headers}"))
    print("--------------------")

    try:
        print("\nNavigating to the page...")
        page.goto("http://localhost:8000/index.html", timeout=60000)

        print("\nWaiting for 10 seconds to capture logs...")
        time.sleep(10)

        print("\n--- Captured Browser Console Output ---")
        if not console_messages:
            print("No console messages were captured.")
        else:
            full_console_log = "\n".join(console_messages)
            print(full_console_log)
        print("--------------------------------------")

        # Check for the canary message
        canary_found = any('--- SCRIPT EXECUTION STARTED ---' in msg for msg in console_messages)
        if canary_found:
            print("\nSUCCESS: Canary message found. JavaScript is executing.")
        else:
            print("\nFAILURE: Canary message NOT found. JavaScript is not executing or logs are not being captured.")

        print("\nVerification script completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")
        raise
    finally:
        browser.close()

with sync_playwright() as playwright:
    run_network_test(playwright)