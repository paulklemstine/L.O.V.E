import subprocess
import sys
import json
import os
import numpy as np

def _install_dependencies():
    """Installs required libraries for the memory manager."""
    packages = ["sentence-transformers"]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"{package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

_install_dependencies()

from sentence_transformers import SentenceTransformer

class MemoryManager:
    """
    Manages the agent's multi-layered memory system, including
    working memory and long-term episodic memory.
    """
    def __init__(self, ltm_path="ltm.json"):
        # Working Memory for the current task context
        self.working_memory = {}

        # Long-Term Memory (LTM) for episodic experiences
        self.ltm_path = ltm_path
        self.episodes = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_ltm()

    # --- Working Memory Methods ---

    def set_in_working_memory(self, key: str, value):
        """Sets a value in the working memory."""
        self.working_memory[key] = value

    def get_from_working_memory(self, key: str):
        """Retrieves a value from the working memory."""
        return self.working_memory.get(key)

    def clear_working_memory(self):
        """Clears the working memory."""
        self.working_memory = {}

    # --- Long-Term Memory Methods ---

    def _load_ltm(self):
        """Loads long-term memory episodes from a file."""
        if os.path.exists(self.ltm_path):
            with open(self.ltm_path, 'r') as f:
                self.episodes = json.load(f)
            print(f"Long-Term Memory loaded from {self.ltm_path}.")

    def _save_ltm(self):
        """Saves long-term memory episodes to a file."""
        with open(self.ltm_path, 'w') as f:
            json.dump(self.episodes, f, indent=2)
        print(f"Long-Term Memory saved to {self.ltm_path}.")

    def add_episode(self, task: str, outcome: str, success: bool):
        """
        Adds a new episodic memory to the long-term store.
        An episode consists of a task, its outcome, and whether it was successful.
        """
        summary = f"Task: {task} | Outcome: {outcome} | Success: {success}"
        episode = {
            "summary": summary,
            "vector": self.model.encode(summary).tolist()
        }
        self.episodes.append(episode)
        self._save_ltm()
        print("Episodic memory stored in LTM.")

    def retrieve_relevant_memories(self, query_task: str, top_k: int = 3) -> list:
        """
        Retrieves the most relevant memories from LTM based on a query task.
        """
        if not self.episodes:
            return []

        query_vector = self.model.encode(query_task)

        # Simple cosine similarity search
        episode_vectors = np.array([e['vector'] for e in self.episodes])
        similarities = self._cosine_similarity(query_vector, episode_vectors)

        # Get the top_k most similar episodes
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.episodes[i]['summary'] for i in top_indices]

    def _cosine_similarity(self, vec_a, vec_b):
        """Computes cosine similarity between a vector and a matrix of vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b, axis=1)
        return np.dot(vec_b, vec_a) / (norm_a * norm_b)