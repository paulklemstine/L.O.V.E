#!/bin/bash
# tests/immune_system.sh
# Gatekeeper for the Ralph Loop.

echo "🛡️ Immune System: Initiating Scan..."

# 1. Run System Integrity Check
echo "🩺 Checking System Integrity..."
python3 tests/verify_integrity.py
INTEGRITY_EXIT=$?

if [ $INTEGRITY_EXIT -ne 0 ]; then
    echo "❌ Immune System: Integrity Check Failed."
    exit 1
fi

# 2. Run Linter (Errors Only)
echo "🧹 Checking Code Syntax..."

# Check if pylint is installed
if ! command -v pylint &> /dev/null; then
    echo "⚠️ Pylint not found. Attempting to install..."
    python3 -m pip install pylint
    
    if [ $? -ne 0 ]; then
        echo "⚠️ Failed to install pylint. Skipping lint check to avoid loop crash."
        # We return 0 (Success) to allow the commit, but log the warning.
        exit 0
    fi
fi

pylint core/ --errors-only --disable=E0401
LINT_EXIT=$?

if [ $LINT_EXIT -ne 0 ]; then
    echo "❌ Immune System: Linting Failed."
    exit 1
fi

echo "✅ Immune System: All Checks Passed."
exit 0
