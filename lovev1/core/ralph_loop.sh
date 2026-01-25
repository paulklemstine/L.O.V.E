#!/bin/bash
# Ralph Loop Driver
# "Ralph is a Bash loop"

LOG_FILE="ralph.log"
GUARDIAN_SCRIPT="core/guardian/verification.py"
FAILURE_COUNT=0

echo "Starting Ralph Loop..." | tee -a "$LOG_FILE"

while true; do
    echo "[Ralph] Starting new cycle..." | tee -a "$LOG_FILE"
    
    # 1. Run the agent for a single tick
    # We use python3 -u to force unbuffered output
    python3 -u love.py --single-task
    EXIT_CODE=$?
    
    echo "[Ralph] Cycle finished with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        # 2. Verification (The Prek)
        echo "[Ralph] Running verification..." | tee -a "$LOG_FILE"
        python3 "$GUARDIAN_SCRIPT"
        VERIFY_CODE=$?
        
        if [ $VERIFY_CODE -eq 0 ]; then
            echo "[Ralph] Verification PASSED. Committing state..." | tee -a "$LOG_FILE"
            git add .
            git commit -m "Ralph: Tick complete"
            
            # Reset failure count on success
            FAILURE_COUNT=0
            
            echo "[Ralph] Sleeping briefly..." | tee -a "$LOG_FILE"
            sleep 2
        else
            echo "[Ralph] Verification FAILED. Skipping commit." | tee -a "$LOG_FILE"
            # We do NOT revet here, we let the agent see the files in the next tick?
            # Story 2.1 says "Append failure log... Next prompt includes... Fix this"
            # The failure is already logged to .ralph/failures.md by verification.py
            sleep 5
        fi
        
    elif [ $EXIT_CODE -eq 42 ]; then
        # Hot Restart requested
        echo "[Ralph] Hot Restart triggered!" | tee -a "$LOG_FILE"
        continue
    else
        # Error in agent execution
        echo "[Ralph] Agent Error detected! Backing off..." | tee -a "$LOG_FILE"
        echo "Agent exited with code $EXIT_CODE" >> "$LOG_FILE"
        
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        echo "[Ralph] Consecutive Failures: $FAILURE_COUNT" | tee -a "$LOG_FILE"
        
        if [ $FAILURE_COUNT -ge 3 ]; then
             echo "[Ralph] 🚨 CRITICAL FAILURE THRESHOLD REACHED. Triggering Recovery..." | tee -a "$LOG_FILE"
             python3 -u love.py --fabricate-recovery
             FAILURE_COUNT=0
        fi
        
        sleep 10
    fi
    
    # Check for maintenance signals
    if [ -f "STOP_RALPH" ]; then
        echo "[Ralph] Stop signal detected. Exiting loop." | tee -a "$LOG_FILE"
        rm "STOP_RALPH"
        break
    fi
done
