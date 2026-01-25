"""
Tests for Story 2.2: Dependency Self-Management
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestScanImports:
    """Tests for scan_imports function."""
    
    def test_scan_imports_basic(self):
        """Test scanning a simple Python file."""
        from core.dependency_manager import scan_imports
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("import requests\nfrom flask import Flask\n")
            
            imports = scan_imports(tmpdir)
            
            assert "requests" in imports
            assert "flask" in imports
    
    def test_scan_imports_excludes_stdlib(self):
        """Test that standard library is excluded."""
        from core.dependency_manager import scan_imports
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("import os\nimport sys\nimport json\nimport requests\n")
            
            imports = scan_imports(tmpdir)
            
            assert "os" not in imports
            assert "sys" not in imports
            assert "json" not in imports
            assert "requests" in imports
    
    def test_scan_imports_handles_syntax_error(self):
        """Test that files with syntax errors are skipped."""
        from core.dependency_manager import scan_imports
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with syntax error
            bad_file = os.path.join(tmpdir, "bad.py")
            with open(bad_file, "w") as f:
                f.write("import requests\nthis is not valid python!!!")
            
            # Should not raise
            imports = scan_imports(tmpdir)
            # May or may not include 'requests' depending on parse behavior


class TestParseRequirements:
    """Tests for parse_requirements function."""
    
    def test_parse_requirements_basic(self):
        """Test parsing simple requirements."""
        from core.dependency_manager import parse_requirements
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("requests==2.28.0\nflask>=2.0.0\n")
            f.flush()
            
            reqs = parse_requirements(f.name)
            
            assert "requests" in reqs
            assert reqs["requests"] == "==2.28.0"
            assert "flask" in reqs
            
            os.unlink(f.name)
    
    def test_parse_requirements_with_comments(self):
        """Test that comments are ignored."""
        from core.dependency_manager import parse_requirements
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# This is a comment\nrequests==2.28.0\n")
            f.flush()
            
            reqs = parse_requirements(f.name)
            
            assert len(reqs) == 1
            assert "requests" in reqs
            
            os.unlink(f.name)


class TestCompareRequirements:
    """Tests for compare_requirements function."""
    
    def test_compare_finds_missing(self):
        """Test detection of missing packages."""
        from core.dependency_manager import compare_requirements
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("flask==2.0.0\n")
            f.flush()
            
            result = compare_requirements({"requests", "flask"}, f.name)
            
            assert "requests" in result["missing"]
            assert "flask" not in result["missing"]
            
            os.unlink(f.name)
    
    def test_compare_finds_unused(self):
        """Test detection of unused packages."""
        from core.dependency_manager import compare_requirements
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("flask==2.0.0\ndjango==4.0.0\n")
            f.flush()
            
            result = compare_requirements({"flask"}, f.name)
            
            assert "django" in result["unused"]
            
            os.unlink(f.name)


class TestImportToPackageMap:
    """Tests for import name to package name mapping."""
    
    def test_yaml_maps_to_pyyaml(self):
        """Test yaml -> pyyaml mapping."""
        from core.dependency_manager import IMPORT_TO_PACKAGE_MAP
        
        assert IMPORT_TO_PACKAGE_MAP.get("yaml") == "pyyaml"
    
    def test_pil_maps_to_pillow(self):
        """Test PIL -> Pillow mapping."""
        from core.dependency_manager import IMPORT_TO_PACKAGE_MAP
        
        assert IMPORT_TO_PACKAGE_MAP.get("PIL") == "Pillow"


class TestSyncRequirements:
    """Tests for sync_requirements function."""
    
    @patch('core.dependency_manager.get_pip_executable')
    @patch('subprocess.run')
    def test_sync_creates_backup(self, mock_run, mock_pip):
        """Test that backup is created before sync."""
        from core.dependency_manager import sync_requirements
        
        mock_pip.return_value = ["pip"]
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file and requirements
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("import requests\n")
            
            req_file = os.path.join(tmpdir, "requirements.txt")
            with open(req_file, "w") as f:
                f.write("flask==2.0.0\n")
            
            result = sync_requirements(tmpdir, req_file, auto_install=False)
            
            # Check backup was created
            assert os.path.exists(f"{req_file}.bak")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
