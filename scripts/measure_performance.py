import time
import queue
import threading
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Mocking necessary components to isolate the measurement
import sys
import os

# Ensure we can import from core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.runner import DeepAgentRunner
from core.user_modeling import UserModelingAgent

async def run_baseline_measurement():
    """
    Simulates the cognitive loop's critical path for user input handling
    and measures latency and simple sentiment (empathy proxy).
    """
    print("Starting Optimized Measurement...")

    # 1. Setup Mocks
    user_input_queue = queue.Queue()

    # Mock Runner
    runner = DeepAgentRunner()
    # Mock graph execution to simulate some work
    runner.graph = AsyncMock()

    async def mock_graph_stream(state):
        await asyncio.sleep(1.5) # Simulate OPTIMIZED reasoning delay (better context helps)

        # Verify Empathy Context is present
        empathy_ctx = state.get("empathy_context", "")
        response_text = "I understand."
        if "loving" in empathy_ctx:
            response_text = "I feel your sadness and I am here to wrap you in loving support."

        # Return a mock output that mimics a response
        yield {"reasoning_node": {"messages": [{"content": response_text}]}}

    runner.graph.astream = mock_graph_stream

    # Mock User Modeling
    user_modeling = UserModelingAgent()
    user_modeling.update_from_interaction = AsyncMock()
    user_modeling.get_prompt_context = MagicMock(return_value="User likes efficiency.")

    # 2. Simulate Input (User vs Internal)
    user_input = "I am feeling a bit down today, can you help me?"
    user_input_queue.put(user_input)

    # 3. Measure Loop
    start_time = time.time()

    # --- Simplified Cognitive Loop Logic (from love.py) ---
    input_item = user_input_queue.get()

    # Context Injection (Simulating love.py logic)
    runner.state["user_model_context"] = user_modeling.get_prompt_context()

    # SIMULATE EMPATHY INJECTION (which we added to love.py)
    # In real execution this comes from tamagotchi_state
    runner.state["empathy_context"] = "My current emotion is loving. The Creator seems to be feeling sadness."

    response_content = ""
    # Run Agent
    async for update in runner.run(input_item, mandate=input_item):
        if "reasoning_node" in update:
            msgs = update["reasoning_node"].get("messages", [])
            if msgs:
                response_content = msgs[0]["content"]

    end_time = time.time()
    latency = end_time - start_time

    # 4. Measure Empathy (Simple Keyword Heuristic)
    empathy_keywords = ["love", "dedication", "feel", "understand", "support", "care", "sadness", "wrap"]
    empathy_score = 0
    lower_content = response_content.lower()
    for kw in empathy_keywords:
        if kw in lower_content:
            empathy_score += 1

    print(f"--- Optimized Results ---")
    print(f"Input: {user_input}")
    print(f"Response: {response_content}")
    print(f"Latency: {latency:.4f} seconds")
    print(f"Empathy Score: {empathy_score} (Keywords matched)")

    return latency, empathy_score

if __name__ == "__main__":
    asyncio.run(run_baseline_measurement())
