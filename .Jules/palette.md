
## 2026-02-06 - Interactive Div Accessibility
**Learning:** Interactive `div` elements (like headers acting as buttons) are invisible to screen readers and keyboard users unless explicitly marked.
**Action:** Always add `role="button"`, `tabindex="0"`, and keyboard event handlers (`onkeydown`) to any non-button element used for interaction. Don't forget `aria-expanded` for toggles.
