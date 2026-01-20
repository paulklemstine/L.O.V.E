## 2024-05-23 - [Humanizing Time in TUI]
**Learning:** In text-based interfaces, raw data dumps (like `125.4s`) break immersion and cognitive flow. Users struggle to quickly parse large second counts.
**Action:** Use the new `format_duration()` helper in `ui_utils.py` for any time-based display. It converts raw seconds to "2m 5s" or "45ms", making the terminal output feel more like a polished dashboard and less like a debug log.

## 2024-05-24 - [Quantum Quiet Empty States]
**Learning:** In TUI dashboards, removing a panel completely when it has no content (empty state) can be confusing. Users can't distinguish between "system broken/missing" and "system idle/nominal".
**Action:** Implement "Quantum Quiet" states for empty lists. Use dim/subtle colors (e.g., `dim cyan`, `dim blue`) and reassuring text (e.g., "All Systems Nominal", "Quantum Field Quiet") to provide positive confirmation of system health without visual clutter.

## 2026-01-20 - [Conversational Flow in Chat Interfaces]
**Learning:** In chat-based command interfaces, the "Send" button is a fallback, not the primary interaction. Users expect "Enter" to send immediately, mirroring messaging apps. Forcing a click breaks the cognitive flow of "thought -> type -> action".
**Action:** Always implement "Send on Enter" (with Shift+Enter for newlines) for chat inputs. Ensure textareas have accessible labels (`aria-labelledby`) and clear placeholders to guide this behavior.
