#!/usr/bin/env python3
"""
Verification tests for Epic 3: Self-Evolution & Safety.

Tests Story 3.1 (Surgeon's Sandbox) and Story 3.2 (Rollback Mechanism).
"""
import sys
import os
import tempfile
sys.path.insert(0, '/home/raver1975/L.O.V.E')


def test_imports():
    """Test that all new modules import correctly."""
    print("Testing imports...")
    
    try:
        from core.surgeon.safe_executor import check_syntax, dry_run_import
        print("✅ check_syntax and dry_run_import imports OK")
    except Exception as e:
        print(f"❌ safe_executor imports failed: {e}")
        return False
    
    try:
        from core.version_control import FileBackupManager, run_pytest_verification, apply_patch_with_rollback
        print("✅ FileBackupManager and rollback functions import OK")
    except Exception as e:
        print(f"❌ version_control imports failed: {e}")
        return False
    
    try:
        from core.nodes.evolution_node import evolution_node, verify_evolution_safety
        print("✅ evolution_node imports OK")
    except Exception as e:
        print(f"❌ evolution_node imports failed: {e}")
        return False
    
    return True


def test_syntax_check():
    """Test syntax check functionality."""
    print("\nTesting check_syntax()...")
    
    try:
        from core.surgeon.safe_executor import check_syntax
        
        # Test valid code
        valid_code = "def hello():\n    print('Hello')\n"
        result = check_syntax(valid_code)
        assert result["valid"] == True, "Valid code should pass"
        print("✅ Valid code passes syntax check")
        
        # Test invalid code
        invalid_code = "def bad syntax here:\n"
        result = check_syntax(invalid_code)
        assert result["valid"] == False, "Invalid code should fail"
        assert result["error"] is not None, "Should have error message"
        assert result["line"] is not None, "Should have line number"
        print(f"✅ Invalid code detected: {result['error']} at line {result['line']}")
        
        return True
    except Exception as e:
        print(f"❌ Syntax check failed: {e}")
        return False


def test_dry_run_import():
    """Test dry run import functionality."""
    print("\nTesting dry_run_import()...")
    
    try:
        from core.surgeon.safe_executor import dry_run_import
        
        # Test valid module
        valid_code = "x = 1\ny = 2\nresult = x + y\n"
        result = dry_run_import(valid_code)
        assert result["success"] == True, "Simple module should import"
        assert result["module_loaded"] == True
        print("✅ Valid module imports successfully")
        
        # Test code with missing import (should fail at import time)
        bad_import_code = "import nonexistent_module_xyz123\n"
        result = dry_run_import(bad_import_code)
        assert result["success"] == False, "Bad import should fail"
        assert "ModuleNotFoundError" in result["error_type"] or "ImportError" in str(result)
        print(f"✅ Bad import detected: {result['error_type']}")
        
        return True
    except Exception as e:
        print(f"❌ Dry run import failed: {e}")
        return False


def test_backup_manager():
    """Test FileBackupManager functionality."""
    print("\nTesting FileBackupManager...")
    
    try:
        from core.version_control import FileBackupManager
        
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("original content\n")
            temp_path = f.name
        
        try:
            mgr = FileBackupManager()
            
            # Test backup creation
            backup_path = mgr.create_backup(temp_path)
            assert backup_path is not None, "Backup should be created"
            assert os.path.exists(backup_path), "Backup file should exist"
            print(f"✅ Backup created: {backup_path}")
            
            # Modify original
            with open(temp_path, 'w') as f:
                f.write("modified content\n")
            
            # Test restore
            success = mgr.restore_backup(temp_path)
            assert success == True, "Restore should succeed"
            
            with open(temp_path, 'r') as f:
                content = f.read()
            assert content == "original content\n", "Content should be restored"
            print("✅ Backup restored successfully")
            
            # Test cleanup
            success = mgr.cleanup_backup(temp_path)
            assert success == True
            print("✅ Backup cleaned up")
            
            return True
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)
                
    except Exception as e:
        print(f"❌ Backup manager failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verify_evolution_safety():
    """Test the evolution safety verification."""
    print("\nTesting verify_evolution_safety()...")
    
    try:
        from core.nodes.evolution_node import verify_evolution_safety
        
        # Test safe code
        safe_code = "def safe_function():\n    return 42\n"
        result = verify_evolution_safety(safe_code, "test.py")
        assert result["safe"] == True
        assert "syntax" in result["gates_passed"]
        assert "dry_import" in result["gates_passed"]
        print(f"✅ Safe code verified: gates={result['gates_passed']}")
        
        # Test unsafe code (syntax error)
        unsafe_code = "def broken(\n"
        result = verify_evolution_safety(unsafe_code, "test.py")
        assert result["safe"] == False
        assert result["error"] is not None
        print(f"✅ Unsafe code detected: {result['error']}")
        
        return True
    except Exception as e:
        print(f"❌ Evolution safety check failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Epic 3: Self-Evolution & Safety - Verification")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_syntax_check()
    all_passed &= test_dry_run_import()
    all_passed &= test_backup_manager()
    all_passed &= test_verify_evolution_safety()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All verification tests PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED")
        sys.exit(1)
