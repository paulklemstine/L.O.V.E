"""
Tests for Story 2.1: Auto-Rollback (Immune System)
"""
import pytest
from unittest.mock import patch, MagicMock
import subprocess


class TestCrashLoopDetection:
    """Tests for crash loop detection."""
    
    def test_detect_crash_loop_empty_history(self):
        """Test with empty exit code history."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        assert monitor.detect_crash_loop() is False
    
    def test_detect_crash_loop_insufficient_codes(self):
        """Test with fewer codes than threshold."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        assert monitor.detect_crash_loop([1, 1]) is False
    
    def test_detect_crash_loop_all_failures(self):
        """Test detection when all recent codes are non-zero."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        assert monitor.detect_crash_loop([1, 1, 1]) is True
        assert monitor.detect_crash_loop([0, 1, 1, 1]) is True  # Last 3 are failures
    
    def test_detect_crash_loop_mixed_codes(self):
        """Test with mixed success/failure codes."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        assert monitor.detect_crash_loop([1, 0, 1]) is False
        assert monitor.detect_crash_loop([0, 0, 0]) is False


class TestExceptionLoopDetection:
    """Tests for exception loop detection."""
    
    def test_detect_exception_loop_same_exception(self):
        """Test detection when same exception repeats."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        exceptions = ["ValueError: x", "ValueError: x", "ValueError: x"]
        assert monitor.detect_exception_loop(exceptions) is True
    
    def test_detect_exception_loop_different_exceptions(self):
        """Test with different exceptions."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        exceptions = ["ValueError: x", "TypeError: y", "KeyError: z"]
        assert monitor.detect_exception_loop(exceptions) is False


class TestRecordExitCode:
    """Tests for exit code recording."""
    
    def test_record_exit_code_adds_to_history(self):
        """Test that exit codes are recorded."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=3)
        monitor.record_exit_code(0)
        monitor.record_exit_code(1)
        
        assert len(monitor.exit_code_history) == 2
        assert monitor.exit_code_history == [0, 1]
    
    def test_record_exit_code_limits_history(self):
        """Test that history is limited to threshold*2."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        monitor = SystemIntegrityMonitor(crash_threshold=2)
        for i in range(10):
            monitor.record_exit_code(i)
        
        # Should only keep last 4 (threshold * 2)
        assert len(monitor.exit_code_history) == 4


class TestAutoRollback:
    """Tests for auto-rollback functionality."""
    
    @patch('core.system_integrity_monitor.subprocess.run')
    @patch('core.system_integrity_monitor.open', create=True)
    def test_auto_rollback_success(self, mock_open, mock_run):
        """Test successful rollback."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        # Mock git commands
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n"),  # git rev-parse
            MagicMock(returncode=0),  # git branch
            MagicMock(returncode=0),  # git revert
            MagicMock(returncode=0),  # git commit
        ]
        mock_open.return_value.__enter__ = MagicMock()
        mock_open.return_value.__exit__ = MagicMock()
        
        monitor = SystemIntegrityMonitor()
        result = monitor.auto_rollback("Test failure")
        
        assert result["success"] is True
        assert result["commit_reverted"] == "abc123"
    
    @patch('core.system_integrity_monitor.subprocess.run')
    def test_auto_rollback_no_commit(self, mock_run):
        """Test rollback when getting commit fails."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        
        monitor = SystemIntegrityMonitor()
        result = monitor.auto_rollback("Test failure")
        
        assert result["success"] is False
        assert "Failed to get" in result["message"]


class TestEvolutionLogLogging:
    """Tests for EVOLUTION_LOG.md logging."""
    
    @patch('core.system_integrity_monitor.open', create=True)
    def test_log_failure_writes_entry(self, mock_open):
        """Test that failure is logged to file."""
        from core.system_integrity_monitor import SystemIntegrityMonitor
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        
        monitor = SystemIntegrityMonitor()
        monitor.log_failure_to_evolution_log("abc123def456", "Test reason", "", True)
        
        # Check that write was called
        mock_file.write.assert_called_once()
        written = mock_file.write.call_args[0][0]
        assert "abc123de" in written  # 8 chars of commit
        assert "✅ Reverted" in written


class TestGlobalInstance:
    """Tests for global monitor instance."""
    
    def test_system_monitor_exists(self):
        """Test that global instance is available."""
        from core.system_integrity_monitor import system_monitor
        
        assert system_monitor is not None
        assert isinstance(system_monitor, type(system_monitor))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
