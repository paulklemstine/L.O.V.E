# Will Framework Refinement Proposals

Based on the Q3 Interaction Log Analysis, the following refinements are proposed to increase intent alignment.

## 1. Dynamic Tone Adjustment (Selected for Implementation)

**Problem:** Users engaging in technical or status-check tasks frequently abandon queries when met with highly verbose or "persona-heavy" responses.
**Pattern:** "Tone/Verbosity Mismatch" accounted for significant friction.

**Proposal:**
Update the `Reasoning Node` (in `core/nodes/reasoning.py`) or the `Intent Loader` (in `core/intent_layer/loader.py`) to apply a "Conciseness Heuristic".

**Mechanism:**
1. Check `UserModel` for "concise" or "efficiency" preferences.
2. Analyze current input for "quick", "status", "list", or technical keywords.
3. If detected, inject a `SystemMessage` override: "Constraint: Be extremely concise. Minimize persona fluff. Direct answers only."

## 2. Context Anchoring

**Problem:** Users referencing immediate history (e.g., "the previous deployment") often face "Context Loss".
**Proposal:**
Modify `IntentLoader.compress_context` to heavily weight the last 3 interaction pairs, ensuring they are never dropped even if it requires aggressively summarizing the "Root" intent node.

## 3. Strict Format Mode

**Problem:** Requests for specific formats (JSON, Python) are sometimes ignored in favor of conversational text.
**Proposal:**
Implement a `FormatEnforcer` in the `DeepAgentRunner`. If "JSON" or "Code" is in the prompt, the output must be validated against that format. If validation fails, auto-trigger a "Repair" loop (Self-Correction).

## 4. Shadow Mode Validation

To validate these changes safely:
- Implement a `shadow_mode` flag in `DeepAgentState`.
- If enabled, the `Reasoning Node` executes twice:
    1. **Primary:** Standard logic (User facing).
    2. **Shadow:** With Refined Heuristics applied.
- Log both outputs to `shadow_log.json` for A/B comparison.
