import os
from playwright.sync_api import sync_playwright

def verify_ux():
    file_path = os.path.abspath("love_interface.html")
    file_url = f"file://{file_path}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(file_url)

        # Verify Label
        # It's visually hidden, but should be in the DOM
        label = page.locator("label[for='input-area']")
        if label.count() > 0:
            print("✅ Label found")
            print(f"Label text: {label.inner_text()}")
        else:
            print("❌ Label not found")

        # Verify Placeholder
        textarea = page.locator("#input-area")
        placeholder = textarea.get_attribute("placeholder")
        if placeholder == "Enter command... (Ctrl+Enter to send)":
            print("✅ Placeholder correct")
        else:
            print(f"❌ Placeholder incorrect: {placeholder}")

        # Verify ARIA label on button
        button = page.locator("#send-button")
        aria_label = button.get_attribute("aria-label")
        if aria_label == "Send Command":
            print("✅ Button ARIA label correct")
        else:
            print(f"❌ Button ARIA label incorrect: {aria_label}")

        # Screenshot with focus
        textarea.focus()
        os.makedirs("verification", exist_ok=True)
        page.screenshot(path="verification/focus_state.png")
        print("✅ Screenshot taken: verification/focus_state.png")

        browser.close()

if __name__ == "__main__":
    verify_ux()
