
import json
from collections import Counter
import os
import re

LOG_FILE = ".Jules/agent_and_user_utterances.jsonl"

def analyze_agent_activity():
    """
    Analyzes the agent activity log to provide insights into tool usage and
    interaction counts.
    """
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at '{LOG_FILE}'.")
        print("Please ensure you are running this script from the repository root.")
        return

    total_utterances = 0
    agent_utterances = 0
    tool_calls = Counter()
    agent_tool_usage = Counter()

    print(f"Analyzing log file: {LOG_FILE}...")

    with open(LOG_FILE, 'r') as f:
        for line in f:
            total_utterances += 1
            try:
                utterance = json.loads(line)

                if utterance.get('role') == 'agent':
                    agent_utterances += 1
                    if 'tool_code' in utterance:
                        # Use regex to find tool calls, which is more robust
                        code = utterance['tool_code']
                        # Regex to find words followed by an opening parenthesis
                        tool_matches = re.findall(r'(\b\w+)\s*\(', code)

                        for tool_name in tool_matches:
                            tool_calls[tool_name] += 1
                            agent_tool_usage[utterance.get('agent_name', 'Unknown')] += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
                continue

    print("\n--- Agent Activity Analysis ---")
    print(f"Total Utterances: {total_utterances}")
    print(f"Agent Utterances: {agent_utterances}")
    print("-" * 30)

    if tool_calls:
        print("\n🔧 Most Frequently Used Tools:")
        for tool, count in tool_calls.most_common(10):
            print(f"  - {tool}: {count} times")
    else:
        print("\n🔧 No tool calls found.")

    print("-" * 30)

    if agent_tool_usage:
        print("\n🤖 Agent Tool Usage:")
        for agent, count in agent_tool_usage.most_common(10):
            print(f"  - {agent}: {count} tool calls")
    else:
        print("\n🤖 No agent tool usage data found.")

    print("\nAnalysis complete.✨")


if __name__ == "__main__":
    analyze_agent_activity()
