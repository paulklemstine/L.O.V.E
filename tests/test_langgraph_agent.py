import sys
import os
import logging

# Add the project root to the path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agent_graph import app, AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_agent_graph():
    print("Starting Agent Graph Test...")
    initial_state = {
        "goal": "Research LangGraph",
        "messages": [],
        "tools_output": {},
        "current_step": "start",
        "final_answer": None
    }
    
    try:
        for output in app.stream(initial_state):
            for key, value in output.items():
                print(f"Node '{key}':")
                print(f"  State update: {value}")
                print("---")
        print("Test completed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        # If it's an import error, it might be because dependencies aren't installed.
        if isinstance(e, ImportError):
            print("Please ensure langgraph and langchain are installed.")

if __name__ == "__main__":
    test_agent_graph()
