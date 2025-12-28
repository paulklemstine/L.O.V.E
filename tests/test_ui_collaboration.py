"""
Tests for Chapter 6: UI & Collaboration
"""
import pytest


class TestAbortController:
    def test_check_abort_initial(self):
        from core.abort_controller import check_abort, reset_abort
        reset_abort()
        assert check_abort() == False
    
    def test_request_and_check(self):
        from core.abort_controller import request_abort, check_abort, reset_abort
        reset_abort()
        request_abort("test")
        assert check_abort() == True


class TestFileWatcher:
    def test_compute_file_hash(self):
        from core.file_watcher import compute_file_hash
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("test content")
            path = f.name
        
        h = compute_file_hash(path)
        assert len(h) == 32  # MD5 hex


class TestThoughtChain:
    def test_add_step(self):
        from core.thought_chain import ThoughtChain
        
        chain = ThoughtChain("Test")
        nid = chain.add_step("Step 1", "thinking")
        
        assert nid in chain.nodes
    
    def test_to_mermaid(self):
        from core.thought_chain import ThoughtChain
        
        chain = ThoughtChain()
        chain.add_step("Start", "success")
        
        mermaid = chain.to_mermaid()
        
        assert "flowchart TD" in mermaid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
