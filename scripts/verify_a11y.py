from html.parser import HTMLParser
import sys
import os

print(f"Checking file: core/web/static/index.html")

class A11yVerifier(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found_header = False
        self.attributes = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            attrs_dict = dict(attrs)
            if attrs_dict.get('id') == 'logs-header':
                self.found_header = True
                self.attributes = attrs_dict

verifier = A11yVerifier()
try:
    with open('core/web/static/index.html', 'r') as f:
        verifier.feed(f.read())
except FileNotFoundError:
    print("❌ ERROR: core/web/static/index.html not found")
    sys.exit(1)

if not verifier.found_header:
    print("❌ ERROR: #logs-header not found")
    sys.exit(1)

required = {
    'role': 'button',
    'tabindex': '0',
    'aria-expanded': 'false',
    'onkeydown': 'handleHeaderKey(event)'
}

failed = False
for attr, expected in required.items():
    val = verifier.attributes.get(attr)
    if val != expected:
        print(f"❌ Missing or incorrect {attr}: expected '{expected}', got '{val}'")
        failed = True
    else:
        print(f"✅ Found {attr}='{val}'")

if failed:
    print("⚠️ Verification failed (as expected before changes)")
    sys.exit(1)
else:
    print("🎉 All accessibility checks passed for #logs-header!")
    sys.exit(0)
