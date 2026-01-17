import json
from collections import Counter

def analyze_intent():
    try:
        with open("interaction_logs.json", "r") as f:
            logs = json.load(f)
    except FileNotFoundError:
        print("Logs not found.")
        return

    total = len(logs)
    failures = [l for l in logs if l["explicit_feedback"] == "negative" or l["alignment_score"] < 0.5]

    failure_types = Counter()
    for f in failures:
        # Heuristic: analyze user's last message for clues
        last_msg = f["interactions"][-1]["content"].lower()
        if "too long" in last_msg or "verbosity" in f["detected_intent"]:
            failure_types["Tone/Verbosity Mismatch"] += 1
        elif "json" in last_msg or "format" in f["detected_intent"]:
            failure_types["Format Instruction Violation"] += 1
        elif "status of what" in f["interactions"][-2]["content"].lower() or "context" in f["detected_intent"]:
            failure_types["Context Loss"] += 1
        else:
            failure_types["Other Misalignment"] += 1

    report = f"""# Will Framework Alignment Report

## Executive Summary
Analysis of {total} interaction logs from the past quarter reveals a {len(failures)/total:.0%} misalignment rate.

## Recurring Misalignment Scenarios
The following patterns were identified where the 'Will' framework's contextual weighting or response generation failed to align with user intent:

"""
    for reason, count in failure_types.most_common():
        report += f"- **{reason}**: {count} occurrences\n"
        report += f"  - *Impact*: High user frustration, clarification loops.\n"

        if reason == "Tone/Verbosity Mismatch":
            report += "  - *Root Cause*: Over-weighting of 'persona' prompts vs. 'efficiency' constraints in technical queries.\n"
        elif reason == "Format Instruction Violation":
             report += "  - *Root Cause*: Weak instruction following for output formats when creative prompts are active.\n"
        elif reason == "Context Loss":
             report += "  - *Root Cause*: Context window truncation or aggressive compression dropping immediate history.\n"

        report += "\n"

    report += """## Recommendations
1. **Dynamic Tone Adjustment**: Implement a heuristic to detect "quick" or "status" queries and down-weight verbose persona traits.
2. **Context Anchoring**: Increase priority of immediate interaction history in the context stack.
3. **Strict Format Mode**: Add a check for "JSON" or "Code" keywords to enforce strict output formats.
"""

    with open("alignment_report.md", "w") as f:
        f.write(report)

    print("Generated alignment_report.md")

if __name__ == "__main__":
    analyze_intent()
