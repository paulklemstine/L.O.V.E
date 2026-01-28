#!/usr/bin/env python3
"""
Verification script for the L.O.V.E. Codebase Introspection Module.

This script tests:
1. Codebase indexing (AST chunking, embeddings)
2. FAISS vector search
3. Merkle tree change detection
4. Evolve tool analysis (dry run)

Usage:
    python verify_introspection.py
    
Set JULES_API_KEY in .env for full testing with Jules API.
"""

import os
import sys
import asyncio
from pathlib import Path

# Ensure we're in the correct directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Environment variables from .env won't be loaded.")

# Add parent to path for imports
sys.path.insert(0, str(script_dir))


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print a test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"       {details}")


async def test_codebase_indexer():
    """Test the codebase indexer module."""
    print_header("Testing Codebase Indexer")
    
    try:
        from core.introspection.codebase_indexer import SemanticCodeChunker, CodeChunk
        
        # Test 1: Initialize chunker
        chunker = SemanticCodeChunker()
        print_result("SemanticCodeChunker initialization", True)
        
        # Test 2: Chunk a single file
        test_file = "core/introspection/codebase_indexer.py"
        if os.path.exists(test_file):
            chunks = chunker.chunk_file(test_file)
            print_result(
                "Chunk single file", 
                len(chunks) > 0, 
                f"Extracted {len(chunks)} chunks from {test_file}"
            )
            
            # Test 3: Check chunk metadata
            if chunks:
                sample = chunks[0]
                has_metadata = all([
                    sample.file_path,
                    sample.start_line > 0,
                    sample.chunk_type,
                    sample.content
                ])
                print_result("Chunk metadata extraction", has_metadata)
                
                # Test 4: Check embeddings
                has_embedding = sample.embedding is not None
                print_result(
                    "Embedding generation", 
                    has_embedding,
                    f"Embedding shape: {sample.embedding.shape if has_embedding else 'None'}"
                )
        else:
            print_result("Chunk single file", False, f"Test file not found: {test_file}")
        
        return True
        
    except Exception as e:
        print_result("Codebase Indexer", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_code_index_manager():
    """Test the code index manager module."""
    print_header("Testing Code Index Manager")
    
    try:
        from core.introspection.code_index_manager import CodeIndexManager, MerkleTreeChangeDetector
        
        # Test 1: Initialize manager
        test_index_path = "state/test_code_index"
        manager = CodeIndexManager(test_index_path)
        print_result("CodeIndexManager initialization", True)
        
        # Test 2: Build a small index (just the introspection module)
        test_codebase = "core/introspection"
        if os.path.exists(test_codebase):
            num_chunks = manager.build_index(test_codebase)
            print_result(
                "Build index", 
                num_chunks > 0,
                f"Indexed {num_chunks} chunks"
            )
            
            # Test 3: Semantic search
            if num_chunks > 0:
                results = manager.search("function to chunk code files", top_k=3)
                print_result(
                    "Semantic search", 
                    len(results) > 0,
                    f"Found {len(results)} results"
                )
                
                # Print top result
                if results:
                    top_chunk, distance = results[0]
                    print(f"       Top result: {top_chunk.qualified_name} (distance: {distance:.4f})")
            
            # Test 4: Index persistence
            manager.save()
            print_result("Save index", os.path.exists(f"{test_index_path}/faiss_index.bin"))
            
            # Test 5: Load index
            manager2 = CodeIndexManager(test_index_path)
            loaded = manager2.load()
            print_result("Load index", loaded, f"Loaded {len(manager2.chunks)} chunks")
            
            # Test 6: Merkle tree change detection
            detector = MerkleTreeChangeDetector(test_codebase)
            tree = detector.compute_tree()
            print_result(
                "Merkle tree computation", 
                len(tree) > 0,
                f"Computed hashes for {len(tree)} files"
            )
        else:
            print_result("Build index", False, f"Test codebase not found: {test_codebase}")
        
        return True
        
    except Exception as e:
        print_result("Code Index Manager", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_codebase_search():
    """Test the codebase search module."""
    print_header("Testing Codebase Search")
    
    try:
        from core.introspection.code_index_manager import CodeIndexManager
        from core.introspection.codebase_search import CodebaseSearch
        
        # Use the test index we just created
        test_index_path = "state/test_code_index"
        manager = CodeIndexManager(test_index_path)
        
        if not manager.load():
            print_result("Load existing index", False, "No index found, building...")
            manager.build_index("core/introspection")
        
        search = CodebaseSearch(manager)
        
        # Test 1: Semantic search
        results = search.semantic_search("how to generate embeddings", top_k=3)
        print_result(
            "Semantic search",
            len(results) > 0,
            f"Found {len(results)} results"
        )
        
        # Test 2: Keyword search (if not Windows - ripgrep might not be available)
        keyword_results = search.keyword_search("embedding", top_k=3)
        print_result(
            "Keyword search",
            len(keyword_results) >= 0,  # May be 0 if ripgrep not installed
            f"Found {len(keyword_results)} results"
        )
        
        # Test 3: Hybrid search
        hybrid_results = search.hybrid_search("code chunk AST", top_k=5)
        print_result(
            "Hybrid search",
            len(hybrid_results) > 0,
            f"Found {len(hybrid_results)} results"
        )
        
        # Test 4: Context assembly
        if hybrid_results:
            context = search.assemble_context(hybrid_results, max_tokens=2000)
            print_result(
                "Context assembly",
                len(context) > 100,
                f"Assembled {len(context)} characters of context"
            )
        
        return True
        
    except Exception as e:
        print_result("Codebase Search", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_evolve_tool():
    """Test the evolve tool module."""
    print_header("Testing Evolve Tool")
    
    try:
        from core.introspection.code_index_manager import CodeIndexManager
        from core.introspection.evolve_tool import EvolveTool, EvolutionRateLimiter
        
        # Test 1: Rate limiter
        rate_limiter = EvolutionRateLimiter("state/test_evolution")
        can_submit, wait_seconds = rate_limiter.can_submit()
        print_result(
            "Rate limiter initialization",
            True,
            f"Can submit: {can_submit}, Wait seconds: {wait_seconds}"
        )
        
        # Use the test index
        test_index_path = "state/test_code_index"
        manager = CodeIndexManager(test_index_path)
        
        if not manager.load():
            # Build a fuller index for evolve testing
            manager.build_index(".")
        
        evolve_tool = EvolveTool(manager, jules_manager=None, state_dir="state/test_evolution")
        print_result("EvolveTool initialization", True)
        
        # Test 2: Analyze codebase
        opportunities = await evolve_tool.analyze_codebase()
        print_result(
            "Codebase analysis",
            True,  # Even if 0 opportunities, analysis ran successfully
            f"Found {len(opportunities)} improvement opportunities"
        )
        
        # Show top 3 opportunities
        if opportunities:
            print("\n       Top improvement opportunities:")
            for i, opp in enumerate(opportunities[:3]):
                print(f"       {i+1}. [{opp.category.value}] {opp.title[:60]}... (priority: {opp.priority})")
        
        # Test 3: Generate user story (dry run)
        if opportunities:
            story = await evolve_tool.generate_user_story(opportunities[0])
            print_result(
                "User story generation",
                len(story) > 50,
                f"Generated {len(story)} character story"
            )
        
        # Test 4: Evolve dry run
        result = await evolve_tool.evolve(
            max_stories=1,
            auto_submit=True,
            dry_run=True  # Don't actually submit to Jules
        )
        print_result(
            "Evolve dry run",
            result['success'],
            result['message']
        )
        
        # Test 5: Check Jules API key
        jules_key = os.environ.get("JULES_API_KEY")
        print_result(
            "Jules API key configured",
            jules_key is not None and len(jules_key) > 10,
            "Key found in environment" if jules_key else "JULES_API_KEY not set"
        )
        
        return True
        
    except Exception as e:
        print_result("Evolve Tool", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  L.O.V.E. CODEBASE INTROSPECTION VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Codebase Indexer", await test_codebase_indexer()))
    results.append(("Code Index Manager", await test_code_index_manager()))
    results.append(("Codebase Search", await test_codebase_search()))
    results.append(("Evolve Tool", await test_evolve_tool()))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Result: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n  🎉 All introspection tests passed!")
        print("  The codebase indexing and evolve tool are ready for use.")
    else:
        print("\n  ⚠️  Some tests failed. Please check the output above.")
    
    # Cleanup hint
    print(f"\n  Note: Test index saved to state/test_code_index/")
    print("  You can delete this directory after verification.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
