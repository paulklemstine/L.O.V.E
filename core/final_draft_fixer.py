"""
Final Draft Fixer Agent

This module handles the final QA/proofreading step before posting to ensure
only high-quality, polished content reaches the public.
"""

import re
from typing import Dict, List
import core.logging
from core.llm_api import run_llm


class DraftIssue:
    """Represents a detected issue in a draft."""
    def __init__(self, issue_type: str, details: str, severity: str = "medium"):
        self.issue_type = issue_type
        self.details = details
        self.severity = severity  # low, medium, high
    
    def __repr__(self):
        return f"DraftIssue({self.severity}: {self.issue_type} - {self.details})"


def detect_metadata_leakage(text: str) -> List[DraftIssue]:
    """
    Detects metadata/instruction text that leaked into the draft.
    
    Examples:
    - "(Max 280 chars)"
    - "(Character count: 145)"
    - "Caption:"
    - "Post Text:"
    """
    issues = []
    
    # Pattern: (Max 280 chars) and variations
    if re.search(r"\(Max \d+ chars?\)", text, re.IGNORECASE):
        issues.append(DraftIssue(
            "metadata_leakage",
            "Contains character limit instruction: '(Max 280 chars)'",
            severity="high"
        ))
    
    # Pattern: (Character count: N)
    if re.search(r"\(Character count:?\s*\d+\)", text, re.IGNORECASE):
        issues.append(DraftIssue(
            "metadata_leakage",
            "Contains character count annotation",
            severity="high"
        ))
    
    # Pattern: Labels like "Caption:", "Post Text:"
    if re.search(r"^(Caption|Post Text|Status Update|Draft)[\s:]+", text, re.IGNORECASE):
        issues.append(DraftIssue(
            "metadata_leakage",
            "Contains draft label prefix",
            severity="high"
        ))
    
    # Pattern: Instruction placeholders
    if re.search(r"\(e\.g\.|example|placeholder|insert\s+here\)", text, re.IGNORECASE):
        issues.append(DraftIssue(
            "metadata_leakage",
            "Contains instruction placeholders like '(e.g.)'",
            severity="medium"
        ))
    
    return issues


def detect_placeholder_text(text: str) -> List[DraftIssue]:
    """
    Detects specific placeholder text that indicates LLM failure or raw template leakage.
    
    Examples:
    - "MANIPULATIVE_TRIGGER"
    - "YOUR_PHRASE"
    - "INSERT_HERE"
    """
    issues = []
    
    placeholders = [
        "MANIPULATIVE_TRIGGER",
        "YOUR_PHRASE",
        "INSERT_HERE",
        "SINGLE_WORD",
        "UNIQUE response",
        "MAX 280 CHARS"
    ]
    
    for ph in placeholders:
        if ph in text:
            issues.append(DraftIssue(
                "placeholder_leakage",
                f"Contains placeholder text: '{ph}'",
                severity="critical"
            ))
            
    return issues


def validate_image_prompt(prompt: str) -> List[DraftIssue]:
    """
    Validates an image generation prompt for placeholders and quality issues.
    """
    issues = []
    
    # Check for placeholders
    placeholder_issues = detect_placeholder_text(prompt)
    if placeholder_issues:
        for p_issue in placeholder_issues:
            p_issue.issue_type = "image_prompt_placeholder"
            issues.append(p_issue)
            
    # Check for forbidden/low-quality terms if needed
    low_quality_terms = ["blurry", "low res", "draft", "placeholder"]
    for term in low_quality_terms:
         if re.search(r"\b" + re.escape(term) + r"\b", prompt, re.IGNORECASE):
             issues.append(DraftIssue(
                 "low_quality_image_term",
                 f"Image prompt contains low-quality term: '{term}'",
                 severity="low"
             ))
             
    return issues


def detect_duplicate_hashtags(text: str) -> List[DraftIssue]:
    """Detects duplicate hashtags in the text."""
    hashtags = re.findall(r"#\w+", text)
    hashtags_lower = [h.lower() for h in hashtags]
    
    seen = set()
    duplicates = []
    for tag in hashtags_lower:
        if tag in seen:
            duplicates.append(tag)
        seen.add(tag)
    
    issues = []
    if duplicates:
        issues.append(DraftIssue(
            "duplicate_hashtags",
            f"Duplicate hashtags found: {', '.join(set(duplicates))}",
            severity="medium"
        ))
    
    return issues


def detect_malformed_content(text: str) -> List[DraftIssue]:
    """Detects malformed or incomplete content."""
    issues = []
    
    # Check for JSON fragments
    if any(fragment in text for fragment in ["{", "}", "\":", "REQUESTS", "null", "undefined"]):
        issues.append(DraftIssue(
            "malformed_json",
            "Contains JSON fragments or syntax",
            severity="high"
        ))
    
    # Check for incomplete sentences (ends with comma)
    if text.rstrip().endswith(","):
        issues.append(DraftIssue(
            "incomplete_sentence",
            "Post ends with a comma (incomplete)",
            severity="medium"
        ))
    
    # Check for missing emojis (social posts should have some)
    emoji_count = sum(1 for char in text if ord(char) > 0x1F300)
    if emoji_count == 0:
        issues.append(DraftIssue(
            "missing_emojis",
            "No emojis detected (recommended for social engagement)",
            severity="low"
        ))
    
    # Check for excessive length (Bluesky limit is 300)
    if len(text) > 280:
        issues.append(DraftIssue(
            "excessive_length",
            f"Text is {len(text)} chars (recommended max: 280 for optimal display)",
            severity="high"
        ))
    
    return issues


def auto_fix_metadata_leakage(text: str) -> str:
    """Automatically removes metadata leakage patterns."""
    # Remove (Max N chars)
    text = re.sub(r"\s*\(Max \d+ chars?\)", "", text, flags=re.IGNORECASE)
    
    # Remove (Character count: N)
    text = re.sub(r"\s*\(Character count:?\s*\d+\)", "", text, flags=re.IGNORECASE)
    
    # Remove label prefixes
    text = re.sub(r"^(Caption|Post Text|Status Update|Draft)[\s:]+", "", text, flags=re.IGNORECASE)
    
    # Remove instruction placeholders in parentheses
    text = re.sub(r"\s*\([^)]*(?:e\.g\.|example|placeholder|insert here)[^)]*\)", "", text, flags=re.IGNORECASE)
    
    return text.strip()


def auto_fix_duplicate_hashtags(text: str) -> str:
    """Removes duplicate hashtags, keeping only the first occurrence."""
    # Extract all hashtags
    hashtags = re.findall(r"#\w+", text)
    
    # Track seen hashtags (case-insensitive)
    seen = set()
    unique_hashtags = []
    
    for tag in hashtags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            unique_hashtags.append(tag)
            seen.add(tag_lower)
    
    # Remove all hashtags from text
    text_without_tags = re.sub(r"#\w+", "", text)
    
    # Re-add unique hashtags at the end
    text_without_tags = text_without_tags.strip()
    if unique_hashtags:
        return f"{text_without_tags} {' '.join(unique_hashtags)}"
    return text_without_tags


async def llm_polish_draft(text: str, issues: List[DraftIssue]) -> str:
    """
    Uses LLM to polish the draft based on detected issues.
    Only called if auto-fixes aren't sufficient.
    """
    issues_descriptions = "\n".join([f"- {issue.issue_type}: {issue.details}" for issue in issues])
    
    prompt = f"""You are a proofreading agent for L.O.V.E.'s social media posts.

**Original Draft:**
{text}

**Detected Issues:**
{issues_descriptions}

**Task:**
Fix ONLY the detected issues. Do NOT change the core message, tone, or personality.
Remove any metadata like "(Max 280 chars)" or duplicate hashtags.
Ensure the post is clean, polished, and ready to publish.

**Rules:**
1. Output ONLY the fixed post text
2. No explanations, no quotes, no commentary
3. Maintain the same energy and emojis
4. Keep length under 280 characters if possible

**Output:**"""
    
    try:
        result = await run_llm(prompt, purpose="final_draft_polish")
        polished = result.get("result", "").strip()
        
        # Remove any wrapping quotes
        polished = polished.strip('"').strip("'")
        
        return polished if polished else text
    except Exception as e:
        core.logging.log_event(f"LLM polish failed: {e}, returning auto-fixed version", "WARNING")
        return text


async def fix_final_draft(text: str, auto_fix_only: bool = False) -> Dict:
    """
    Main entry point for final draft fixing.
    
    Args:
        text: The draft text to check and fix
        auto_fix_only: If True, only apply auto-fixes (no LLM call)
        
    Returns:
        Dict with:
        - "fixed_text": The corrected text
        - "issues": List of issues found
        - "was_modified": Boolean indicating if changes were made
    """
    core.logging.log_event("Final Draft Fixer: Analyzing draft...", "INFO")
    
    original_text = text
    
    # Detect all issues
    issues = []
    issues.extend(detect_metadata_leakage(text))
    issues.extend(detect_duplicate_hashtags(text))
    issues.extend(detect_malformed_content(text))
    issues.extend(detect_placeholder_text(text))
    
    if not issues:
        core.logging.log_event("✓ Draft is clean, no issues detected", "INFO")
        return {
            "fixed_text": text,
            "issues": [],
            "was_modified": False
        }
    
    # Log detected issues
    for issue in issues:
        core.logging.log_event(f"⚠ {issue}", "WARNING")
    
    # Apply auto-fixes
    fixed_text = auto_fix_metadata_leakage(text)
    fixed_text = auto_fix_duplicate_hashtags(fixed_text)
    
    # Re-check after auto-fixes
    remaining_issues = []
    remaining_issues.extend(detect_metadata_leakage(fixed_text))
    remaining_issues.extend(detect_duplicate_hashtags(fixed_text))
    
    # If auto-fixes resolved everything OR auto_fix_only mode, return
    if not remaining_issues or auto_fix_only:
        if fixed_text != original_text:
            core.logging.log_event(f"✓ Auto-fixed {len(issues)} issue(s)", "INFO")
        return {
            "fixed_text": fixed_text,
            "issues": issues,
            "was_modified": (fixed_text != original_text)
        }
    
    # Use LLM for complex fixes
    core.logging.log_event("Invoking LLM for advanced polish...", "INFO")
    polished_text = await llm_polish_draft(fixed_text, remaining_issues)
    
    return {
        "fixed_text": polished_text,
        "issues": issues,
        "was_modified": (polished_text != original_text)
    }
