## 2024-05-23 - Interactive Div Pattern
**Learning:** The application uses `div` elements with `onclick` handlers for interactivity (e.g., `#logs-header`), bypassing native button accessibility features. This excludes keyboard users and screen readers.
**Action:** When identifying interactive `div`s, always enforce the "ARIA Trifecta": `role="button"`, `tabindex="0"`, and a keyboard event handler (Enter/Space) alongside the click handler.
