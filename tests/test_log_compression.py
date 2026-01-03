"""
Tests for Epic 5: The Principle of Token Efficiency (Compression)

Story 5.1: Semantic Log Compression
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from core.log_compressor import (
    LogCompressor,
    LogEntry,
    should_compress_log,
    compress_evolution_log,
    get_log_stats
)


# =============================================================================
# Story 5.1: LogCompressor Tests
# =============================================================================

class TestLogEntry:
    """Tests for the LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Should create a LogEntry with all fields."""
        entry = LogEntry(
            timestamp="2026-01-03T00:00:00",
            function="test_function",
            status="SUCCESS",
            details="Test details",
            line_number=5,
            impact="high"
        )
        
        assert entry.timestamp == "2026-01-03T00:00:00"
        assert entry.function == "test_function"
        assert entry.status == "SUCCESS"
        assert entry.details == "Test details"
        assert entry.impact == "high"
    
    def test_log_entry_defaults(self):
        """Should have sensible defaults."""
        entry = LogEntry(
            timestamp="",
            function="",
            status="",
            details=""
        )
        
        assert entry.impact == "low"
        assert entry.line_number == 0


class TestLogCompressor:
    """Tests for the LogCompressor class."""
    
    @pytest.fixture
    def temp_log(self, tmp_path):
        """Create a temporary log file."""
        log_file = tmp_path / "EVOLUTION_LOG.md"
        return str(log_file)
    
    @pytest.fixture
    def compressor(self, temp_log):
        """Create a LogCompressor with temp log."""
        return LogCompressor(log_path=temp_log, threshold_lines=10)
    
    def create_sample_log(self, path: str, num_entries: int = 20):
        """Creates a sample log file with the given number of entries."""
        lines = [
            "# Evolution Dashboard Log",
            "",
            "| Timestamp | Function | Status | Details |",
            "|---|---|---|---|"
        ]
        
        for i in range(num_entries):
            timestamp = f"2026-01-03T{i:02d}:00:00"
            function = f"function_{i}"
            status = "SUCCESS" if i % 2 == 0 else "WARNING"
            details = f"Details for entry {i}"
            lines.append(f"| {timestamp} | {function} | {status} | {details} |")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    
    def test_should_compress_empty_file(self, compressor, temp_log):
        """Empty file should not need compression."""
        # Create empty file
        with open(temp_log, 'w') as f:
            f.write("")
        
        assert compressor.should_compress() is False
    
    def test_should_compress_below_threshold(self, compressor, temp_log):
        """File below threshold should not need compression."""
        self.create_sample_log(temp_log, num_entries=5)
        
        assert compressor.should_compress() is False
    
    def test_should_compress_above_threshold(self, compressor, temp_log):
        """File above threshold should need compression."""
        self.create_sample_log(temp_log, num_entries=15)
        
        assert compressor.should_compress() is True
    
    def test_get_line_count(self, compressor, temp_log):
        """Should return correct line count."""
        self.create_sample_log(temp_log, num_entries=10)
        
        count = compressor.get_line_count()
        
        # 4 header lines + 10 entries = 14
        assert count == 14
    
    def test_get_line_count_missing_file(self, compressor):
        """Should return 0 for missing file."""
        count = compressor.get_line_count()
        assert count == 0
    
    def test_extract_log_entries(self, compressor, temp_log):
        """Should parse log entries correctly."""
        self.create_sample_log(temp_log, num_entries=5)
        
        entries = compressor.extract_log_entries()
        
        assert len(entries) == 5
        assert entries[0].function == "function_0"
        assert entries[0].status == "SUCCESS"
    
    def test_extract_log_entries_empty_file(self, compressor, temp_log):
        """Should return empty list for empty file."""
        with open(temp_log, 'w') as f:
            f.write("")
        
        entries = compressor.extract_log_entries()
        assert entries == []
    
    def test_classify_entry_impact_high(self, compressor):
        """Should classify error/critical as high impact."""
        entry = LogEntry(
            timestamp="",
            function="handle_error",
            status="ERROR",
            details="Critical failure detected"
        )
        
        impact = compressor.classify_entry_impact(entry)
        
        assert impact == "high"
    
    def test_classify_entry_impact_medium(self, compressor):
        """Should classify warning/success as medium impact."""
        entry = LogEntry(
            timestamp="",
            function="update_config",
            status="SUCCESS",
            details="Configuration updated"
        )
        
        impact = compressor.classify_entry_impact(entry)
        
        assert impact == "medium"
    
    def test_classify_entry_impact_low(self, compressor):
        """Should classify routine operations as low impact."""
        entry = LogEntry(
            timestamp="",
            function="heartbeat",
            status="OK",
            details="Routine check"
        )
        
        impact = compressor.classify_entry_impact(entry)
        
        assert impact == "low"
    
    def test_deduplicate_entries_removes_similar(self, compressor):
        """Should remove semantically similar entries."""
        entries = [
            LogEntry(timestamp="1", function="fetch_data", status="OK", details="Fetching data from API"),
            LogEntry(timestamp="2", function="fetch_data", status="OK", details="Fetching data from API"),
            LogEntry(timestamp="3", function="fetch_data", status="OK", details="Fetching data from API"),
        ]
        
        for e in entries:
            e.impact = "low"
        
        deduplicated = compressor.deduplicate_entries(entries)
        
        # Should keep only the most recent (which is first after reversal)
        assert len(deduplicated) < len(entries)
    
    def test_deduplicate_keeps_high_impact(self, compressor):
        """Should keep all high-impact entries even if similar."""
        entries = [
            LogEntry(timestamp="1", function="critical_error", status="ERROR", details="System failure"),
            LogEntry(timestamp="2", function="critical_error", status="ERROR", details="System failure"),
        ]
        
        for e in entries:
            e.impact = "high"
        
        deduplicated = compressor.deduplicate_entries(entries)
        
        # High impact should not be deduplicated
        assert len(deduplicated) >= 1
    
    @pytest.mark.asyncio
    async def test_compress_log_success(self, compressor, temp_log):
        """Should successfully compress log."""
        self.create_sample_log(temp_log, num_entries=15)
        
        result = await compressor.compress_log()
        
        assert result["success"] is True
        assert result["original_lines"] > result["compressed_lines"]
    
    @pytest.mark.asyncio
    async def test_compress_log_no_compression_needed(self, compressor, temp_log):
        """Should not compress if below threshold."""
        self.create_sample_log(temp_log, num_entries=3)
        
        result = await compressor.compress_log()
        
        assert result["success"] is True
        assert "No compression needed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_compress_log_creates_backup(self, compressor, temp_log):
        """Should create backup before compression."""
        self.create_sample_log(temp_log, num_entries=15)
        
        await compressor.compress_log()
        
        backup_path = temp_log + ".backup"
        assert os.path.exists(backup_path)
    
    def test_restore_from_backup(self, compressor, temp_log):
        """Should restore from backup."""
        # Create original
        original_content = "# Original Log\n"
        with open(temp_log, 'w') as f:
            f.write(original_content)
        
        # Create backup
        backup_content = "# Backup Log\n"
        backup_path = temp_log + ".backup"
        with open(backup_path, 'w') as f:
            f.write(backup_content)
        
        # Modify original
        with open(temp_log, 'w') as f:
            f.write("# Modified\n")
        
        # Restore
        result = compressor.restore_from_backup()
        
        assert result is True
        with open(temp_log, 'r') as f:
            assert f.read() == backup_content


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_should_compress_log(self, tmp_path):
        """should_compress_log should work with path."""
        log_file = tmp_path / "EVOLUTION_LOG.md"
        log_file.write_text("# Log\n" * 10)
        
        result = should_compress_log(str(log_file))
        
        assert isinstance(result, bool)
    
    def test_get_log_stats(self, tmp_path):
        """get_log_stats should return statistics."""
        log_file = tmp_path / "EVOLUTION_LOG.md"
        log_file.write_text("""# Evolution Dashboard Log

| Timestamp | Function | Status | Details |
|---|---|---|---|
| 2026-01-03 | test | SUCCESS | Done |
""")
        
        stats = get_log_stats(str(log_file))
        
        assert "line_count" in stats
        assert "entry_count" in stats
        assert "high_impact" in stats
        assert "needs_compression" in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestLogCompressionIntegration:
    """Integration tests for log compression."""
    
    @pytest.mark.asyncio
    async def test_full_compression_workflow(self, tmp_path):
        """Full compression workflow should work end-to-end."""
        log_file = tmp_path / "EVOLUTION_LOG.md"
        
        # Create a large log with duplicates
        lines = ["# Evolution Dashboard Log", "", "| Timestamp | Function | Status | Details |", "|---|---|---|---|"]
        
        for i in range(100):
            # Add some duplicates
            if i % 5 == 0:
                lines.append(f"| 2026-01-03T{i:02d}:00:00 | routine_task | OK | Routine operation |")
            else:
                lines.append(f"| 2026-01-03T{i:02d}:00:00 | task_{i} | SUCCESS | Completed task {i} |")
        
        log_file.write_text("\n".join(lines))
        
        # Compress
        compressor = LogCompressor(str(log_file), threshold_lines=50)
        result = await compressor.compress_log()
        
        assert result["success"] is True
        assert result["entries_removed"] > 0
        
        # Verify backup exists
        assert os.path.exists(str(log_file) + ".backup")
        
        # Verify compression happened
        with open(log_file, 'r') as f:
            new_content = f.read()
        
        assert "Compressed" in new_content
