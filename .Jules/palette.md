# Palette's Journal

## 2024-05-22 - Accessibility of Interactive Divs
**Learning:** Interactive elements implemented as `div`s (like the logs header) are invisible to screen readers and keyboard users unless they have explicit `role`, `tabindex`, and keyboard handlers.
**Action:** Always wrap interactive `div`s with `role="button"`, `tabindex="0"`, and ensure `Enter`/`Space` keys trigger the same action as click.
