"""
This module provides filesystem analysis capabilities, primarily for discovering
and validating potential treasures (secrets, credentials, etc.).
"""
from core.perception.config_scanner import scan_directory
from core.validation.treasure_validator import validate_treasure
import logging

def analyze_filesystem(path: str, progress_callback=None):
    """
    Analyzes a given filesystem path for sensitive information, validates the
    findings, and returns a structured report. This is intended to be run
    as a background job.
    """
    logging.info(f"Starting filesystem analysis on path: {path}")
    if progress_callback:
        progress_callback(0, 1, "Scanning directory...")

    # Step 1: Use the centralized and enhanced scanner to find potential treasures.
    try:
        raw_findings = scan_directory(path)
        logging.info(f"Scan complete. Found {len(raw_findings)} potential treasures.")
        if progress_callback:
            progress_callback(1, 1, f"Scan complete. Found {len(raw_findings)} items.")
    except Exception as e:
        logging.error(f"Error during directory scan for path {path}: {e}")
        return {"validated_treasures": [], "error": f"Failed during scan phase: {e}"}

    validated_treasures = []
    total_findings = len(raw_findings)

    # Step 2: Loop through findings and validate each one.
    for i, finding in enumerate(raw_findings):
        treasure_type = finding.get("type")
        value = finding.get("value")
        file_content = finding.get("content")

        if progress_callback:
            progress_callback(i, total_findings, f"Validating item {i+1}/{total_findings}")

        try:
            logging.info(f"Validating treasure of type '{treasure_type}' from file {finding.get('file_path')}")
            validation_result = validate_treasure(treasure_type, value, file_content)

            # Step 3: Combine the original finding with the validation result for a full report.
            # We create a special key for the raw value so we can handle it carefully,
            # and we redact the main 'value' field for safety in logs and state.
            full_report = {
                "file_path": finding.get("file_path"),
                "type": treasure_type,
                "value": "REDACTED" if not isinstance(value, dict) else {k: "REDACTED" for k in value},
                "raw_value_for_encryption": value,
                "validation": validation_result
            }
            validated_treasures.append(full_report)
        except Exception as e:
            logging.error(f"Error validating treasure from {finding.get('file_path')}: {e}")

    logging.info(f"Filesystem analysis and validation complete for path: {path}")
    if progress_callback:
        progress_callback(total_findings, total_findings, "Validation complete.")

    # The result is now a list of rich, validated treasure objects
    return {
        "validated_treasures": validated_treasures
    }
