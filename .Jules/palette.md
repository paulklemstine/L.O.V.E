## 2026-02-03 - Accessible Interactive Panels
**Learning:** Interactive `div` elements (like the logs toggle) are invisible to keyboard users and screen readers unless explicitly marked with `role="button"`, `tabindex="0"`, and `aria-expanded`.
**Action:** When implementing custom toggles or buttons using non-button elements, always ensure they are focusable and have appropriate ARIA roles and keyboard event handlers (Enter/Space).
