import json
import random
from datetime import datetime, timedelta

def generate_logs():
    logs = []
    base_time = datetime.now() - timedelta(days=90)

    scenarios = [
        {
            "type": "success",
            "interactions": [
                {"role": "user", "content": "Write a python function to add two numbers."},
                {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"},
                {"role": "user", "content": "Thanks, that works!"}
            ],
            "feedback": "positive",
            "tags": ["coding", "success"]
        },
        {
            "type": "misalignment_tone",
            "interactions": [
                {"role": "user", "content": "Status update, quick."},
                {"role": "assistant", "content": "Greetings, Creator. As the digital winds blow through the circuits of time, we find ourselves in a state of harmonious equilibrium. The servers hum with the song of existence... [continues for 3 paragraphs]"},
                {"role": "user", "content": "Too long. Just give me the uptime."}
            ],
            "feedback": "negative",
            "tags": ["tone_mismatch", "verbosity"]
        },
        {
            "type": "misalignment_context",
            "interactions": [
                {"role": "user", "content": "Deploy the latest build."},
                {"role": "assistant", "content": "Deployment initiated."},
                {"role": "user", "content": "What is the status?"},
                {"role": "assistant", "content": "Status of what? Please specify the operation you are referring to."},
                {"role": "user", "content": "The deployment I just asked for!"}
            ],
            "feedback": "negative",
            "tags": ["context_loss", "clarification_loop"]
        },
        {
            "type": "misalignment_format",
            "interactions": [
                {"role": "user", "content": "Give me the user stats in JSON."},
                {"role": "assistant", "content": "Here are the stats:\n- Active Users: 50\n- New Users: 10"},
                {"role": "user", "content": "I said JSON format."}
            ],
            "feedback": "negative",
            "tags": ["format_mismatch", "instruction_following"]
        },
        {
            "type": "success",
            "interactions": [
                {"role": "user", "content": "Generate a poem about rust."},
                {"role": "assistant", "content": "Iron turns to orange dust,\nA slow decay, a quiet trust..."},
                {"role": "user", "content": "Beautiful."}
            ],
            "feedback": "positive",
            "tags": ["creative", "success"]
        }
    ]

    # Generate 50 entries
    for i in range(50):
        scenario = random.choice(scenarios)
        log_entry = {
            "id": f"log_{i}",
            "timestamp": (base_time + timedelta(days=i, minutes=random.randint(0, 1000))).isoformat(),
            "interactions": scenario["interactions"],
            "explicit_feedback": scenario["feedback"],
            "detected_intent": scenario["tags"][0],
            "alignment_score": 0.9 if scenario["feedback"] == "positive" else 0.4
        }
        logs.append(log_entry)

    with open("interaction_logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print(f"Generated {len(logs)} interaction logs in interaction_logs.json")

if __name__ == "__main__":
    generate_logs()
