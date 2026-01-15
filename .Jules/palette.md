## 2024-05-23 - [Humanizing Time in TUI]
**Learning:** In text-based interfaces, raw data dumps (like `125.4s`) break immersion and cognitive flow. Users struggle to quickly parse large second counts.
**Action:** Use the new `format_duration()` helper in `ui_utils.py` for any time-based display. It converts raw seconds to "2m 5s" or "45ms", making the terminal output feel more like a polished dashboard and less like a debug log.

## 2024-05-24 - [Quantum Quiet Empty States]
**Learning:** In TUI dashboards, removing a panel completely when it has no content (empty state) can be confusing. Users can't distinguish between "system broken/missing" and "system idle/nominal".
**Action:** Implement "Quantum Quiet" states for empty lists. Use dim/subtle colors (e.g., `dim cyan`, `dim blue`) and reassuring text (e.g., "All Systems Nominal", "Quantum Field Quiet") to provide positive confirmation of system health without visual clutter.

## 2025-01-28 - [Retrofitting A11y to Div-Buttons]
**Learning:** Adding `role="button"` to a `div` is not enough. Without `tabindex="0"` and `keydown` listeners (for Enter/Space), the element remains invisible to keyboard users, creating a "fake accessible" state that fails compliance checks.
**Action:** When converting `div`s to buttons, always implement the "Trinity of Interactivity": `role="button"`, `tabindex="0"`, and `keydown` handler.
