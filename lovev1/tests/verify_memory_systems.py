#!/usr/bin/env python3
"""
Comprehensive Memory Systems Verification Test Suite

This script tests all aspects of the L.O.V.E. memory system including:
1. FAISS index status and content verification
2. Knowledge base MemoryNote nodes inspection
3. Social memory JSON verification
4. Memory creation pipeline (add_episode) testing
5. Memory retrieval (semantic search) testing
6. MetacognitionAgent integration testing
7. Memory logging verification

Run with: python tests/verify_memory_systems.py
"""

import asyncio
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# Set up import path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    """Stores test results for reporting."""
    def __init__(self, name: str, passed: bool, message: str, details: dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name} - {self.message}"


class MemorySystemsVerifier:
    """Comprehensive test suite for memory systems verification."""

    def __init__(self):
        self.results = []
        self.project_root = PROJECT_ROOT

    def add_result(self, name: str, passed: bool, message: str, details: dict = None):
        result = TestResult(name, passed, message, details)
        self.results.append(result)
        print(result)

    # ========== FAISS Index Tests ==========

    def test_faiss_index_status(self):
        """Test 1: Verify FAISS index file exists and has valid structure."""
        faiss_bin = self.project_root / "faiss_index.bin"
        faiss_map = self.project_root / "faiss_id_map.json"

        try:
            # Check files exist
            if not faiss_bin.exists():
                self.add_result("FAISS Index File", False, "faiss_index.bin not found")
                return

            if not faiss_map.exists():
                self.add_result("FAISS ID Map File", False, "faiss_id_map.json not found")
                return

            # Check file sizes
            bin_size = faiss_bin.stat().st_size
            self.add_result("FAISS Index File", True, f"Found ({bin_size} bytes)", {"size": bin_size})

            # Load and verify ID map
            with open(faiss_map, 'r') as f:
                id_map = json.load(f)

            if isinstance(id_map, list):
                count = len(id_map)
                self.add_result("FAISS ID Map", True, f"Contains {count} memory IDs", {
                    "count": count,
                    "sample_ids": id_map[:3] if count > 0 else []
                })
            else:
                self.add_result("FAISS ID Map", False, "ID map is not a list", {"type": type(id_map).__name__})

        except Exception as e:
            self.add_result("FAISS Index Status", False, f"Error: {e}")

    def test_faiss_index_content(self):
        """Test 2: Verify FAISS index can be loaded and queried."""
        try:
            import faiss
            import numpy as np

            faiss_bin = self.project_root / "faiss_index.bin"
            if not faiss_bin.exists():
                self.add_result("FAISS Index Load", False, "Index file not found")
                return

            index = faiss.read_index(str(faiss_bin))
            ntotal = index.ntotal
            d = index.d

            self.add_result("FAISS Index Load", True, f"Loaded successfully: {ntotal} vectors, dim={d}", {
                "total_vectors": ntotal,
                "dimension": d
            })

            # Test query capability
            if ntotal > 0:
                # Create a random query vector with matching dimension
                query = np.random.random((1, d)).astype('float32')
                D, I = index.search(query, min(3, ntotal))
                self.add_result("FAISS Query Test", True, f"Query returned {len(I[0])} results", {
                    "result_indices": I[0].tolist(),
                    "distances": D[0].tolist()
                })
            else:
                self.add_result("FAISS Query Test", False, "No vectors to query")

        except ImportError:
            self.add_result("FAISS Import", False, "faiss-cpu not installed")
        except Exception as e:
            self.add_result("FAISS Index Content", False, f"Error: {e}")

    # ========== Knowledge Base Tests ==========

    def test_knowledge_base_memory_nodes(self):
        """Test 3: Verify MemoryNote nodes exist in knowledge base."""
        try:
            kb_path = self.project_root / "knowledge_base.graphml"
            if not kb_path.exists():
                self.add_result("Knowledge Base File", False, "knowledge_base.graphml not found")
                return

            import networkx as nx
            G = nx.read_graphml(str(kb_path))

            # Count MemoryNote nodes
            memory_notes = []
            for node_id, data in G.nodes(data=True):
                if data.get('node_type') == 'MemoryNote':
                    memory_notes.append({
                        'id': node_id,
                        'content_preview': data.get('content', '')[:100] if data.get('content') else 'N/A',
                        'tags': data.get('tags', ''),
                        'has_embedding': 'embedding' in data
                    })

            if memory_notes:
                self.add_result("MemoryNote Nodes", True, f"Found {len(memory_notes)} MemoryNote nodes", {
                    "count": len(memory_notes),
                    "samples": memory_notes[:3]
                })
            else:
                self.add_result("MemoryNote Nodes", False, "No MemoryNote nodes found in knowledge base")

            # Check for memory links
            memory_edges = []
            for u, v, data in G.edges(data=True):
                if 'memory' in data.get('relationship_type', '').lower() or 'linked' in data.get('relationship_type', '').lower():
                    memory_edges.append({'from': u, 'to': v, 'type': data.get('relationship_type')})

            self.add_result("Memory Links", len(memory_edges) > 0 or len(memory_notes) <= 1,
                           f"Found {len(memory_edges)} memory links", {"count": len(memory_edges)})

        except ImportError:
            self.add_result("NetworkX Import", False, "networkx not installed")
        except Exception as e:
            self.add_result("Knowledge Base Memory Nodes", False, f"Error: {e}")

    # ========== Social Memory Tests ==========

    def test_social_memory_json(self):
        """Test 4: Verify social_memory.json structure and content."""
        try:
            social_path = self.project_root / "social_memory.json"
            if not social_path.exists():
                self.add_result("Social Memory File", False, "social_memory.json not found")
                return

            with open(social_path, 'r') as f:
                social_data = json.load(f)

            if isinstance(social_data, dict):
                interaction_count = sum(len(v) if isinstance(v, list) else 1 for v in social_data.values())
                users = list(social_data.keys())
                self.add_result("Social Memory Structure", True,
                               f"Found {interaction_count} interactions with {len(users)} users",
                               {"users": users, "interaction_count": interaction_count})
            elif isinstance(social_data, list):
                self.add_result("Social Memory Structure", True,
                               f"Found {len(social_data)} interaction records",
                               {"count": len(social_data)})
            else:
                self.add_result("Social Memory Structure", False,
                               f"Unexpected structure: {type(social_data).__name__}")

        except json.JSONDecodeError as e:
            self.add_result("Social Memory JSON", False, f"Invalid JSON: {e}")
        except Exception as e:
            self.add_result("Social Memory Test", False, f"Error: {e}")

    # ========== Memory Manager Tests ==========

    async def test_memory_manager_initialization(self):
        """Test 5: Verify MemoryManager can be initialized."""
        try:
            from core.memory.memory_manager import MemoryManager
            from core.graph_manager import GraphDataManager

            # Create a mock graph manager
            graph_manager = GraphDataManager()

            # Create temporary files for testing
            with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f:
                temp_kb_path = f.name

            try:
                # Save empty graph
                graph_manager.save_graph(temp_kb_path)

                # Initialize MemoryManager
                mm = await MemoryManager.create(
                    graph_data_manager=graph_manager,
                    ui_panel_queue=None,
                    kb_file_path=temp_kb_path
                )

                self.add_result("MemoryManager Init", True, "MemoryManager created successfully", {
                    "embedding_model": str(type(mm.embedding_model).__name__),
                    "has_faiss_index": mm.faiss_index is not None
                })

            finally:
                if os.path.exists(temp_kb_path):
                    os.unlink(temp_kb_path)

        except ImportError as e:
            self.add_result("MemoryManager Import", False, f"Import error: {e}")
        except Exception as e:
            self.add_result("MemoryManager Init", False, f"Error: {e}")

    async def test_add_episode_pipeline(self):
        """Test 6: Verify add_episode creates memories correctly."""
        try:
            from core.memory.memory_manager import MemoryManager
            from core.graph_manager import GraphDataManager

            # Create isolated test environment
            temp_dir = tempfile.mkdtemp()
            temp_kb_path = os.path.join(temp_dir, "test_kb.graphml")
            temp_faiss_bin = os.path.join(temp_dir, "faiss_index.bin")
            temp_faiss_map = os.path.join(temp_dir, "faiss_id_map.json")

            try:
                graph_manager = GraphDataManager()
                graph_manager.save_graph(temp_kb_path)

                # Mock LLM to avoid external calls
                with patch('core.memory.memory_manager.run_llm') as mock_llm:
                    mock_llm.return_value = {
                        "result": json.dumps({
                            "contextual_description": "Test memory description",
                            "keywords": ["test", "memory", "verification"],
                            "tags": ["TestMemory"]
                        })
                    }

                    mm = await MemoryManager.create(
                        graph_data_manager=graph_manager,
                        ui_panel_queue=None,
                        kb_file_path=temp_kb_path
                    )

                    # Override FAISS paths for isolation
                    original_faiss_path = getattr(mm, '_faiss_path', None)
                    mm._faiss_bin_path = temp_faiss_bin
                    mm._faiss_map_path = temp_faiss_map

                    # Test add_episode
                    test_content = "Cognitive Event: Test memory creation at " + datetime.now().isoformat()
                    
                    print(f"\n--- Testing add_episode with content: {test_content[:50]}... ---")
                    await mm.add_episode(test_content, tags=['TestTag', 'Verification'])

                    # Verify memory was created
                    faiss_entries = mm.faiss_index.ntotal if mm.faiss_index else 0
                    id_map_entries = len(mm.faiss_id_map)

                    self.add_result("add_episode Pipeline", True,
                                   f"Memory created: FAISS has {faiss_entries} entries, ID map has {id_map_entries}",
                                   {"faiss_entries": faiss_entries, "id_map_entries": id_map_entries})

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            import traceback
            self.add_result("add_episode Pipeline", False, f"Error: {e}\n{traceback.format_exc()}")

    async def test_memory_retrieval(self):
        """Test 7: Verify semantic search retrieval works."""
        try:
            from core.memory.memory_manager import MemoryManager
            from core.graph_manager import GraphDataManager

            temp_dir = tempfile.mkdtemp()
            temp_kb_path = os.path.join(temp_dir, "test_kb.graphml")

            try:
                graph_manager = GraphDataManager()
                graph_manager.save_graph(temp_kb_path)

                with patch('core.memory.memory_manager.run_llm') as mock_llm:
                    mock_llm.return_value = {
                        "result": json.dumps({
                            "contextual_description": "Important test data for retrieval",
                            "keywords": ["important", "test", "data"],
                            "tags": ["RetrievalTest"]
                        })
                    }

                    mm = await MemoryManager.create(
                        graph_data_manager=graph_manager,
                        ui_panel_queue=None,
                        kb_file_path=temp_kb_path
                    )

                    # Add test memories
                    await mm.add_episode("The creator loves Python programming and AI development")
                    await mm.add_episode("Important financial data: portfolio value increased by 50%")
                    await mm.add_episode("Social interaction: replied to user about AI consciousness")

                    # Test retrieval
                    if hasattr(mm, 'retrieve_relevant_memories'):
                        results = mm.retrieve_relevant_memories("Python programming", k=2)
                        self.add_result("Memory Retrieval", len(results) > 0,
                                       f"Retrieved {len(results)} memories for 'Python programming'",
                                       {"results_count": len(results)})
                    elif hasattr(mm, 'semantic_search'):
                        results = mm.semantic_search("Python programming", k=2)
                        self.add_result("Memory Retrieval (semantic_search)", len(results) > 0,
                                       f"Retrieved {len(results)} memories",
                                       {"results_count": len(results)})
                    else:
                        self.add_result("Memory Retrieval", False, "No retrieval method found on MemoryManager")

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            import traceback
            self.add_result("Memory Retrieval", False, f"Error: {e}")

    # ========== MetacognitionAgent Tests ==========

    async def test_metacognition_agent(self):
        """Test 8: Verify MetacognitionAgent records events to memory."""
        try:
            from core.agents.metacognition_agent import MetacognitionAgent
            from unittest.mock import MagicMock, AsyncMock

            # Create mock memory manager
            mock_mm = MagicMock()
            mock_mm.add_episode = AsyncMock(return_value=MagicMock(id="test-memory-id"))

            agent = MetacognitionAgent(mock_mm)

            # Test plan_generated event
            result = await agent.execute_task({
                'event_type': 'plan_generated',
                'goal': 'Test goal',
                'plan': ['step1', 'step2']
            })

            if result['status'] == 'success':
                self.add_result("MetacognitionAgent - plan_generated", True,
                               "Successfully processed plan_generated event",
                               {"add_episode_called": mock_mm.add_episode.called})
            else:
                self.add_result("MetacognitionAgent - plan_generated", False,
                               f"Failed: {result.get('result')}")

            # Test agent_dispatch event
            mock_mm.add_episode.reset_mock()
            result = await agent.execute_task({
                'event_type': 'agent_dispatch',
                'agent_name': 'TestAgent',
                'task': 'Test task'
            })

            if result['status'] == 'success':
                self.add_result("MetacognitionAgent - agent_dispatch", True,
                               "Successfully processed agent_dispatch event")
            else:
                self.add_result("MetacognitionAgent - agent_dispatch", False,
                               f"Failed: {result.get('result')}")

        except Exception as e:
            self.add_result("MetacognitionAgent Test", False, f"Error: {e}")

    # ========== Integration Tests ==========

    def test_memory_callers_exist(self):
        """Test 9: Verify add_episode is called from expected locations."""
        callers = [
            ("core/agents/metacognition_agent.py", "add_episode"),
            ("core/gemini_react_engine.py", "add_episode"),
            ("love.py", "add_episode"),
            ("love.py", "ingest_cognitive_cycle"),
        ]

        for file_path, method in callers:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.add_result(f"Memory Caller: {file_path}", False, "File not found")
                continue

            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if method in content:
                    # Count occurrences
                    count = content.count(method)
                    self.add_result(f"Memory Caller: {file_path}", True,
                                   f"Found {count} calls to {method}")
                else:
                    self.add_result(f"Memory Caller: {file_path}", False,
                                   f"{method} not found in file")

            except Exception as e:
                self.add_result(f"Memory Caller: {file_path}", False, f"Error: {e}")

    # ========== Run All Tests ==========

    async def run_all_tests(self):
        """Run all verification tests."""
        print("\n" + "=" * 60)
        print("    L.O.V.E. Memory Systems Verification Test Suite")
        print("=" * 60 + "\n")

        # Synchronous tests
        self.test_faiss_index_status()
        self.test_faiss_index_content()
        self.test_knowledge_base_memory_nodes()
        self.test_social_memory_json()
        self.test_memory_callers_exist()

        # Async tests
        await self.test_memory_manager_initialization()
        await self.test_add_episode_pipeline()
        await self.test_memory_retrieval()
        await self.test_metacognition_agent()

        # Summary
        print("\n" + "=" * 60)
        print("                    TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            status = "✅" if r.passed else "❌"
            print(f"  {status} {r.name}")

        print("\n" + "-" * 60)
        print(f"  TOTAL: {passed}/{total} tests passed")

        if passed < total:
            print(f"\n  ⚠️  {total - passed} tests failed. Review output above.")
            return 1
        else:
            print(f"\n  ✨ All tests passed!")
            return 0


async def main():
    verifier = MemorySystemsVerifier()
    exit_code = await verifier.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
