
import pytest
from unittest import mock
import os
from core.guardian.safety import SafetyNet

@pytest.fixture
def safety():
    return SafetyNet()

@pytest.fixture
def temp_file(tmp_path):
    f = tmp_path / "target.py"
    f.write_text("content", encoding="utf-8")
    return str(f)

def test_check_clean_state_clean(safety, temp_file):
    # Mock subprocess.run to return empty stdout (clean)
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        
        assert safety.check_clean_state(temp_file)
        
        # Verify call args
        args, kwargs = mock_run.call_args
        assert args[0] == ["git", "status", "--porcelain", "target.py"]

def test_check_clean_state_dirty(safety, temp_file):
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "M target.py\n" # Dirty
        
        assert not safety.check_clean_state(temp_file)

def test_check_clean_state_error(safety, temp_file):
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 128
        mock_run.return_value.stderr = "Not a git repository"
        
        assert not safety.check_clean_state(temp_file)

def test_backup_restore_flow(safety, temp_file):
    # 1. Create backup
    assert safety.create_backup(temp_file)
    assert os.path.exists(f"{temp_file}.bak")
    
    # 2. Modify original
    with open(temp_file, "w") as f:
        f.write("modified")
        
    # 3. Restore
    assert safety.restore_from_backup(temp_file)
    assert not os.path.exists(f"{temp_file}.bak")
    
    # 4. Verify content restored
    with open(temp_file, "r") as f:
        assert f.read() == "content"

def test_cleanup_backup(safety, temp_file):
    assert safety.create_backup(temp_file)
    assert safety.cleanup_backup(temp_file)
    assert not os.path.exists(f"{temp_file}.bak")

def test_backup_missing_file(safety, tmp_path):
    missing = str(tmp_path / "missing.py")
    assert not safety.create_backup(missing)
