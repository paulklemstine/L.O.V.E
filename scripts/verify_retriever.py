
import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_retriever import ToolRetriever, ToolMatch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyRetriever")

def mock_registry_tools() -> Dict[str, Dict[str, Any]]:
    """Create a mock registry structure with a small number of tools."""
    tools = {}
    for i in range(10):
        name = f"tool_{i}"
        tools[name] = {
            "name": name,
            "description": f"This is tool number {i} for testing purposes.",
            "schema": {
                "name": name,
                "description": f"This is tool number {i} for testing purposes.",
                "parameters": {}
            },
            "searchable": f"tool_{i} testing purposes"
        }
    return tools

def main():
    print("--- Verifying Tool Retriever Fallback ---")
    
    # Initialize retriever
    retriever = ToolRetriever(similarity_threshold=0.5) # High threshold to force failure
    
    # Manually populate cache with mock tools
    print("Populating retriever with 10 mock tools...")
    retriever._tool_cache = mock_registry_tools()
    
    # Query that should definitely fail similarity check
    query = "irrelevant query zingle zangle"
    print(f"Querying for: '{query}'")
    
    matches = retriever.retrieve(query)
    
    print(f"Matches found: {len(matches)}")
    
    # Check if fallback occurred
    if len(matches) == 10:
        print("✅ SUCCESS: Fallback triggered! All tools returned.")
        return 0
    elif len(matches) == 0:
        print("❌ FAILURE: No tools returned. Fallback did NOT trigger.")
        return 1
    else:
        print(f"⚠️ UNEXPECTED: {len(matches)} tools returned.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
