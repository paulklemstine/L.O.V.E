from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:7860/love_interface.html")
    page.screenshot(path="verification.png")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
