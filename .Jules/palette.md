## 2024-05-23 - Accessibility for Custom Interactive Elements
**Learning:** Custom interactive elements (like `div`s used as buttons) are invisible to screen readers and keyboard users unless explicitly marked with ARIA roles and keyboard handlers. Simply adding `onclick` is insufficient.
**Action:** Always pair `onclick` with `onkeydown` (handling Enter/Space), `role="button"`, and `tabindex="0"`. Use `aria-expanded` for collapsible sections.
