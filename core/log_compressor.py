"""
Story 5.1: Semantic Log Compression

This module implements intelligent compression of EVOLUTION_LOG.md to prevent
context window overflow while preserving semantic meaning.

The morphic agent must run indefinitely - information must be compressed, not truncated.
"""

import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from core.logging import log_event


@dataclass
class LogEntry:
    """Represents a single entry in the evolution log."""
    timestamp: str
    function: str
    status: str
    details: str
    line_number: int = 0
    impact: str = "low"  # low, medium, high
    raw_line: str = ""


class LogCompressor:
    """
    Story 5.1: Semantic Log Compression
    
    Compresses EVOLUTION_LOG.md when it exceeds threshold,
    retaining only novel and high-impact events.
    
    Morphic agents must run indefinitely. Context windows are finite.
    Information must be compressed, not just truncated.
    """
    
    # Keywords that indicate high-impact entries
    HIGH_IMPACT_KEYWORDS = [
        "error", "fail", "critical", "exception", "crash", "panic",
        "security", "forbidden", "unauthorized", "blocked",
        "evolution", "mutation", "upgrade", "breakthrough",
        "decision", "choice", "strategy", "goal"
    ]
    
    # Keywords that indicate medium-impact entries
    MEDIUM_IMPACT_KEYWORDS = [
        "warning", "caution", "notice", "important",
        "change", "update", "modify", "create", "delete",
        "success", "complete", "finished", "done"
    ]
    
    def __init__(
        self,
        log_path: str = "EVOLUTION_LOG.md",
        threshold_lines: int = 500,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the LogCompressor.
        
        Args:
            log_path: Path to the evolution log file
            threshold_lines: Trigger compression when exceeding this
            similarity_threshold: Entries above this similarity are duplicates
        """
        self.log_path = log_path
        self.threshold_lines = threshold_lines
        self.similarity_threshold = similarity_threshold
        
        # Get absolute path if relative
        if not os.path.isabs(log_path):
            self.log_path = os.path.join(os.getcwd(), log_path)
    
    def should_compress(self) -> bool:
        """
        Check if the log file exceeds the line threshold.
        
        Returns:
            True if compression should be triggered
        """
        if not os.path.exists(self.log_path):
            return False
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            return line_count > self.threshold_lines
        except Exception as e:
            log_event(f"Error checking log size: {e}", "ERROR")
            return False
    
    def get_line_count(self) -> int:
        """Returns the current line count of the log."""
        if not os.path.exists(self.log_path):
            return 0
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def extract_log_entries(self) -> List[LogEntry]:
        """
        Parses the log file into structured LogEntry objects.
        
        Returns:
            List of LogEntry objects
        """
        if not os.path.exists(self.log_path):
            return []
        
        entries = []
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header lines (title and table header)
            in_table = False
            header_pattern = re.compile(r'^\|.*\|$')
            separator_pattern = re.compile(r'^\|[-|]+\|$')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip header
                if line.startswith('#'):
                    continue
                
                # Skip separator
                if separator_pattern.match(line):
                    in_table = True
                    continue
                
                # Parse table rows
                if in_table and header_pattern.match(line):
                    parts = [p.strip() for p in line.split('|')]
                    # Remove empty strings from split
                    parts = [p for p in parts if p]
                    
                    if len(parts) >= 4:
                        entry = LogEntry(
                            timestamp=parts[0],
                            function=parts[1],
                            status=parts[2],
                            details=parts[3] if len(parts) > 3 else "",
                            line_number=i + 1,
                            raw_line=line
                        )
                        entries.append(entry)
            
        except Exception as e:
            log_event(f"Error parsing log entries: {e}", "ERROR")
        
        return entries
    
    def classify_entry_impact(self, entry: LogEntry) -> str:
        """
        Classifies an entry's impact level.
        
        Args:
            entry: LogEntry to classify
            
        Returns:
            "high", "medium", or "low"
        """
        # Combine all text for analysis
        text = f"{entry.function} {entry.status} {entry.details}".lower()
        
        # Check for high-impact keywords
        for keyword in self.HIGH_IMPACT_KEYWORDS:
            if keyword in text:
                return "high"
        
        # Check for medium-impact keywords
        for keyword in self.MEDIUM_IMPACT_KEYWORDS:
            if keyword in text:
                return "medium"
        
        return "low"
    
    def deduplicate_entries(self, entries: List[LogEntry]) -> List[LogEntry]:
        """
        Removes semantically similar entries, keeping the most recent.
        
        Args:
            entries: List of entries to deduplicate
            
        Returns:
            Deduplicated list of entries
        """
        from core.semantic_similarity import get_similarity_checker
        
        if not entries:
            return entries
        
        checker = get_similarity_checker()
        deduplicated = []
        seen_texts = []
        
        # Process in reverse order (most recent first)
        for entry in reversed(entries):
            entry_text = f"{entry.function} {entry.status} {entry.details}"
            
            # Check if too similar to any seen entry
            is_duplicate = False
            for seen in seen_texts:
                similarity = checker.compute_similarity(entry_text, seen)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # High impact entries always kept
                if entry.impact == "high" or not is_duplicate:
                    deduplicated.append(entry)
                    seen_texts.append(entry_text)
        
        # Reverse back to original order
        deduplicated.reverse()
        
        return deduplicated
    
    def _summarize_entries(
        self, 
        entries: List[LogEntry],
        max_entries: int = 100
    ) -> str:
        """
        Creates a compressed summary of log entries.
        
        Args:
            entries: Entries to summarize
            max_entries: Maximum entries to include in summary
            
        Returns:
            Markdown formatted summary string
        """
        if not entries:
            return ""
        
        # Group by impact
        high_impact = [e for e in entries if e.impact == "high"]
        medium_impact = [e for e in entries if e.impact == "medium"]
        low_impact = [e for e in entries if e.impact == "low"]
        
        lines = []
        lines.append("# Evolution Dashboard Log (Compressed)")
        lines.append("")
        lines.append(f"*Compressed on {datetime.now().isoformat()}*")
        lines.append(f"*Original entries: {len(entries)}, Kept: {min(len(entries), max_entries)}*")
        lines.append("")
        
        # High impact section
        if high_impact:
            lines.append("## High Impact Events")
            lines.append("")
            lines.append("| Timestamp | Function | Status | Details |")
            lines.append("|---|---|---|---|")
            for entry in high_impact[:max_entries // 2]:
                lines.append(f"| {entry.timestamp} | {entry.function} | {entry.status} | {entry.details} |")
            lines.append("")
        
        # Medium impact (summarized)
        if medium_impact:
            lines.append("## Notable Events")
            lines.append("")
            lines.append("| Timestamp | Function | Status | Details |")
            lines.append("|---|---|---|---|")
            for entry in medium_impact[:max_entries // 3]:
                lines.append(f"| {entry.timestamp} | {entry.function} | {entry.status} | {entry.details} |")
            lines.append("")
        
        # Low impact (highly compressed)
        if low_impact:
            lines.append("## Routine Operations")
            lines.append("")
            lines.append(f"*{len(low_impact)} routine entries compressed*")
            lines.append("")
            # Just show last few
            if low_impact:
                lines.append("| Timestamp | Function | Status | Details |")
                lines.append("|---|---|---|---|")
                for entry in low_impact[-10:]:
                    lines.append(f"| {entry.timestamp} | {entry.function} | {entry.status} | {entry.details} |")
                lines.append("")
        
        return "\n".join(lines)
    
    async def compress_log(self) -> Dict[str, Any]:
        """
        Main compression workflow.
        
        Story 5.1: Compress EVOLUTION_LOG.md when it exceeds threshold.
        
        Workflow:
        1. Parse log into entries
        2. Classify each entry's impact
        3. Deduplicate similar entries
        4. Compress remaining entries
        5. Write to EVOLUTION_LOG_COMPRESSED.md
        6. Replace original
        
        Returns:
            {
                "success": bool,
                "original_lines": int,
                "compressed_lines": int,
                "entries_removed": int,
                "message": str
            }
        """
        result = {
            "success": False,
            "original_lines": 0,
            "compressed_lines": 0,
            "entries_removed": 0,
            "message": ""
        }
        
        try:
            # Get original line count
            result["original_lines"] = self.get_line_count()
            
            if result["original_lines"] <= self.threshold_lines:
                result["message"] = "No compression needed"
                result["success"] = True
                return result
            
            log_event(
                f"🗜️ Starting log compression: {result['original_lines']} lines",
                "INFO"
            )
            
            # Step 1: Parse entries
            entries = self.extract_log_entries()
            if not entries:
                result["message"] = "No entries to compress"
                result["success"] = True
                return result
            
            original_count = len(entries)
            
            # Step 2: Classify impact
            for entry in entries:
                entry.impact = self.classify_entry_impact(entry)
            
            # Step 3: Deduplicate
            deduplicated = self.deduplicate_entries(entries)
            
            # Step 4: Create compressed summary
            compressed_content = self._summarize_entries(deduplicated)
            
            # Step 5: Write compressed file
            compressed_path = self.log_path.replace(".md", "_COMPRESSED.md")
            backup_path = self.log_path + ".backup"
            
            # Backup original
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
            
            # Write compressed
            with open(compressed_path, 'w', encoding='utf-8') as f:
                f.write(compressed_content)
            
            # Step 6: Replace original
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write(compressed_content)
            
            # Calculate results
            result["compressed_lines"] = len(compressed_content.split('\n'))
            result["entries_removed"] = original_count - len(deduplicated)
            result["success"] = True
            result["message"] = (
                f"Compressed from {result['original_lines']} to "
                f"{result['compressed_lines']} lines. "
                f"Removed {result['entries_removed']} duplicate entries."
            )
            
            log_event(
                f"✅ Log compression complete: {result['message']}",
                "INFO"
            )
            
        except Exception as e:
            result["message"] = f"Compression failed: {str(e)}"
            log_event(f"❌ Log compression failed: {e}", "ERROR")
        
        return result
    
    def restore_from_backup(self) -> bool:
        """
        Restores the log from backup if compression caused issues.
        
        Returns:
            True if restored, False if no backup exists
        """
        backup_path = self.log_path + ".backup"
        
        if os.path.exists(backup_path):
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            
            log_event("📂 Restored log from backup", "INFO")
            return True
        
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def should_compress_log(log_path: str = "EVOLUTION_LOG.md") -> bool:
    """Check if the log needs compression."""
    compressor = LogCompressor(log_path)
    return compressor.should_compress()


async def compress_evolution_log(
    log_path: str = "EVOLUTION_LOG.md",
    threshold: int = 500
) -> Dict[str, Any]:
    """
    Convenience function to compress the evolution log.
    
    Args:
        log_path: Path to the log file
        threshold: Line threshold for triggering compression
        
    Returns:
        Compression result dictionary
    """
    compressor = LogCompressor(log_path, threshold_lines=threshold)
    return await compressor.compress_log()


def get_log_stats(log_path: str = "EVOLUTION_LOG.md") -> Dict[str, Any]:
    """Get statistics about the evolution log."""
    compressor = LogCompressor(log_path)
    entries = compressor.extract_log_entries()
    
    impacts = {"high": 0, "medium": 0, "low": 0}
    for entry in entries:
        entry.impact = compressor.classify_entry_impact(entry)
        impacts[entry.impact] += 1
    
    return {
        "line_count": compressor.get_line_count(),
        "entry_count": len(entries),
        "high_impact": impacts["high"],
        "medium_impact": impacts["medium"],
        "low_impact": impacts["low"],
        "needs_compression": compressor.should_compress()
    }
