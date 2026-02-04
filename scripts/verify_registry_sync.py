
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool_registry import ToolRegistry, register_tool, tool_schema
from core.tool_retriever import get_tool_retriever, retrieve_tools_for_step

def dummy_tool_a():
    """I am tool A."""
    pass

def dummy_tool_b():
    """I am tool B."""
    pass

def verify_sync():
    print("--- Verifying Registry-Retriever Sync ---")
    
    # 1. Setup
    registry = ToolRegistry()
    registry.register(tool_schema(dummy_tool_a))
    
    retriever = get_tool_retriever()
    # Clear any previous state
    retriever._tool_cache.clear()
    
    # Enable listener (the fix we are testing)
    retriever.listen_to_registry(registry)
    
    # 2. Retrieve Tool A (forces indexing)
    print("Searching for Tool A...")
    matches = retrieve_tools_for_step("I want tool A", registry=registry)
    found_a = any(m.name == "dummy_tool_a" for m in matches)
    print(f"Found Tool A: {found_a}")
    
    if not found_a:
        print("❌ Failed initial retrieval!")
        return
        
    # 3. Add Tool B to Registry
    print("\nAdding Tool B to Registry...")
    registry.register(tool_schema(dummy_tool_b))
    
    # 4. Search for Tool B WITHOUT manual re-indexing
    # The retriever should hopefully see it, or fail if it's caching hard
    print("Searching for Tool B...")
    matches = retrieve_tools_for_step("I want tool B", registry=registry)
    found_b = any(m.name == "dummy_tool_b" for m in matches)
    
    if found_b:
        print("✅ SUCCESS: Retriever found Tool B!")
    else:
        print("❌ FAILURE: Retriever did NOT find Tool B (Sync Issue)")
        print(f"Cache size: {len(retriever._tool_cache)}")
        print(f"Registry size: {len(registry._tools)}")

if __name__ == "__main__":
    verify_sync()
