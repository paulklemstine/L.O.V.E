import json
import os
import time

class LongTermMemory:
    def __init__(self, memory_file="long_term_memory.json"):
        self.memory_file = memory_file
        self.memories = self._load_memories()
        print("LongTermMemory: Initialized.")

    def _load_memories(self):
        """Loads memories from the JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []

    def save_episodic_memory(self, task, outcome, summary):
        """Saves a memory of a completed task and its outcome."""
        memory_entry = {
            "timestamp": time.time(),
            "task": task,
            "outcome": outcome,
            "summary": summary
        }
        self.memories.append(memory_entry)
        self._persist_memories()
        print(f"LongTermMemory: Saved episodic memory for task '{task}'.")

    def retrieve_similar_memories(self, task_description):
        """
        Retrieves memories of similar past tasks.
        This is a simple keyword-based retrieval and can be improved with vector search.
        """
        keywords = set(task_description.lower().split())
        similar_memories = [
            mem for mem in self.memories
            if keywords.intersection(set(mem["task"].lower().split()))
        ]
        print(f"LongTermMemory: Retrieved {len(similar_memories)} similar memories for task '{task_description}'.")
        return similar_memories

    def _persist_memories(self):
        """Saves the current memories to the JSON file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=4)