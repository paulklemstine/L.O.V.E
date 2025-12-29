#!/usr/bin/env python3
"""
Verification tests for Epic 2: Cognitive Architecture & Memory.

Tests Story 2.1 (Semantic Memory Bridge) and Story 2.2 (Memory Folding Strategy).
"""
import sys
sys.path.insert(0, '/home/raver1975/L.O.V.E')

def test_imports():
    """Test that all new modules import correctly."""
    print("Testing imports...")
    
    try:
        from core.nodes.memory_bridge import memory_bridge_node, format_memory_context_for_prompt
        print("✅ memory_bridge imports OK")
    except Exception as e:
        print(f"❌ memory_bridge failed: {e}")
        return False
    
    try:
        from core.nodes.fold_memory import fold_memory_node, should_trigger_folding
        print("✅ fold_memory imports OK")
    except Exception as e:
        print(f"❌ fold_memory failed: {e}")
        return False
    
    try:
        from core.memory.schemas import KnowledgeNugget, MemorySummary
        print("✅ schemas imports OK")
    except Exception as e:
        print(f"❌ schemas failed: {e}")
        return False
    
    try:
        from core.state import DeepAgentState
        print("✅ state imports OK")
    except Exception as e:
        print(f"❌ state failed: {e}")
        return False
    
    return True


def test_knowledge_nugget():
    """Test KnowledgeNugget schema creation."""
    print("\nTesting KnowledgeNugget schema...")
    
    try:
        from core.memory.schemas import KnowledgeNugget
        
        nugget = KnowledgeNugget(
            content="Test summary of conversation",
            source_message_count=5,
            key_directives=["Must fix bug", "Priority task"],
            topics=["code", "bug"],
            token_savings=150
        )
        
        assert nugget.content == "Test summary of conversation"
        assert nugget.source_message_count == 5
        assert len(nugget.key_directives) == 2
        assert nugget.token_savings == 150
        print("✅ KnowledgeNugget creation works")
        return True
    except Exception as e:
        print(f"❌ KnowledgeNugget failed: {e}")
        return False


def test_state_has_memory_context():
    """Test that DeepAgentState includes memory_context field."""
    print("\nTesting DeepAgentState memory_context field...")
    
    try:
        from core.state import DeepAgentState
        from typing import get_type_hints
        
        hints = get_type_hints(DeepAgentState)
        
        assert 'memory_context' in hints, "memory_context not in state"
        assert 'memory_manager' in hints, "memory_manager not in state"
        print("✅ DeepAgentState has memory_context and memory_manager fields")
        return True
    except Exception as e:
        print(f"❌ State check failed: {e}")
        return False


def test_memory_context_formatting():
    """Test memory context formatting for prompts."""
    print("\nTesting memory context formatting...")
    
    try:
        from core.nodes.memory_bridge import format_memory_context_for_prompt
        
        test_context = [
            {
                "id": "test-1",
                "content": "User asked about fixing a bug in the login system.",
                "contextual_description": "Bug fix request for authentication module.",
                "keywords": ["bug", "login", "authentication"],
                "tags": ["CodeFix", "Priority"]
            },
            {
                "id": "test-2",
                "content": "Previous attempt to fix the bug failed due to missing imports.",
                "contextual_description": "Failed bug fix attempt.",
                "keywords": ["import", "error"],
                "tags": ["Failure"]
            }
        ]
        
        formatted = format_memory_context_for_prompt(test_context)
        
        assert "Relevant Past Interactions" in formatted
        assert "Past Interaction 1" in formatted
        assert "Past Interaction 2" in formatted
        assert "Bug fix request" in formatted
        print("✅ Memory context formatting works")
        return True
    except Exception as e:
        print(f"❌ Formatting failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Epic 2: Cognitive Architecture & Memory - Verification")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_knowledge_nugget()
    all_passed &= test_state_has_memory_context()
    all_passed &= test_memory_context_formatting()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All verification tests PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED")
        sys.exit(1)
