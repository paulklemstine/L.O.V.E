import pytest
import os
from pathlib import Path
import re

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

FRONTEND_PATH = Path("core/web/static/index.html")

def test_logs_header_accessibility():
    """Verify that the logs header has proper accessibility attributes."""
    if not FRONTEND_PATH.exists():
        pytest.fail(f"Frontend file not found at {FRONTEND_PATH}")

    content = FRONTEND_PATH.read_text(encoding="utf-8")

    if HAS_BS4:
        soup = BeautifulSoup(content, "html.parser")
        header = soup.find(id="logs-header")

        assert header is not None, "Could not find #logs-header element"

        # Check attributes
        assert header.get("role") == "button", "Missing or incorrect role='button'"
        assert header.get("tabindex") == "0", "Missing or incorrect tabindex='0'"
        assert header.get("aria-expanded") == "false", "Missing or incorrect aria-expanded='false'"
        assert header.get("aria-controls") == "log-container", "Missing or incorrect aria-controls='log-container'"
        assert "handleHeaderKey(event)" in header.get("onkeydown", ""), "Missing onkeydown handler"

    else:
        # Fallback to string checks
        print("WARNING: BeautifulSoup not installed, using string checks.")
        header_pattern = r'<div[^>]*id="logs-header"[^>]*>'
        match = re.search(header_pattern, content)
        assert match is not None, "Could not find #logs-header tag"

        tag_content = match.group(0)
        assert 'role="button"' in tag_content, "Missing role='button'"
        assert 'tabindex="0"' in tag_content, "Missing tabindex='0'"
        assert 'aria-expanded="false"' in tag_content, "Missing aria-expanded='false'"
        assert 'aria-controls="log-container"' in tag_content, "Missing aria-controls"
        assert 'onkeydown="handleHeaderKey(event)"' in tag_content, "Missing onkeydown handler"

def test_javascript_accessibility_logic():
    """Verify that JS logic includes accessibility updates."""
    content = FRONTEND_PATH.read_text(encoding="utf-8")

    # Check for handleHeaderKey function
    assert "function handleHeaderKey(event)" in content, "Missing handleHeaderKey function"

    # Check that toggleLogs updates aria-expanded
    # Looking for setAttribute('aria-expanded', ...) or similar logic
    # We expect something like: header.setAttribute('aria-expanded', !isCollapsed)
    # Or checking if it toggles.

    # Since we are implementing it, we know what to look for.
    # We will look for logic that touches aria-expanded inside toggleLogs

    toggle_logs_pattern = r"function toggleLogs\(\)\s*\{([^}]*)\}"
    match = re.search(toggle_logs_pattern, content, re.DOTALL)
    assert match is not None, "Could not find toggleLogs function"

    func_body = match.group(1)
    assert "aria-expanded" in func_body, "toggleLogs does not seem to update aria-expanded"
