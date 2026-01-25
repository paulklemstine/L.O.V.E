import threading
import time
from collections import defaultdict

# Mock the MODEL_STATS structure
MODEL_STATS = defaultdict(lambda: {"total_tokens_generated": 0, "total_time_spent": 0.0, "successful_calls": 0, "failed_calls": 0, "reasoning_score": 50.0, "provider": "unknown"})

def modifier_thread():
    """Continuously adds new keys to MODEL_STATS to simulate background updates."""
    for i in range(1000):
        MODEL_STATS[f"new_model_{i}"]  # This triggers defaultdict to add a key
        time.sleep(0.001)

def iterator_thread():
    """Iterates over MODEL_STATS, simulating rank_models."""
    try:
        # Simulate the buggy loop in rank_models
        # BUG: Iterating directly over .items() while it changes size
        for _ in range(100):
            count = 0
            for k, v in list(MODEL_STATS.items()):
                count += 1
                # Simulate some work
                _ = v["reasoning_score"] * 0.5 
            print(f"Iteration finished, count: {count}")
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
        return
    print("No error caught (unexpected if race condition triggered)")

if __name__ == "__main__":
    print("Starting race condition reproduction...")
    # Pre-fill some data
    for i in range(10):
        MODEL_STATS[f"existing_{i}"]
    
    t1 = threading.Thread(target=modifier_thread)
    t2 = threading.Thread(target=iterator_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
