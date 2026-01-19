#!/bin/bash
# scripts/ralph_loop.sh
# The "Ralph" Loop: Autonomy through Iteration and Backpressure

LOG_FILE="logs/ralph.log"
mkdir -p logs

while true; do
    echo "🔄 [$(date)] Ralph is waking up..." | tee -a $LOG_FILE
    
    # 1. Sync Reality
    # git pull origin main >> $LOG_FILE 2>&1
    
    # 2. Execute ONE Atomic Task (Single-Shot Mode)
    # The runner exits after completing one task from PRD or Idle Logic
    # Using --mandate="auto" to let the Orchestrator decide what to do
    python3 love.py --single-task >> $LOG_FILE 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        # 3. Success Path: Verify & Commit
        echo "✅ Task completed. Running Immune System..." | tee -a $LOG_FILE
        
        # Backpressure: Run tests
        if [ -f "tests/immune_system.sh" ]; then
            # Ensure executable
            if [ ! -x "tests/immune_system.sh" ]; then
                echo "⚠️ Making immune system script executable..." | tee -a $LOG_FILE
                chmod +x tests/immune_system.sh
            fi

            echo "🔬 Analyzing system integrity..." | tee -a $LOG_FILE
            
            # Run with output capture
            # We want to see the output in the log AND potentially catch the error
            ./tests/immune_system.sh >> $LOG_FILE 2>&1
            IMMUNE_EXIT=$?
            
            if [ $IMMUNE_EXIT -eq 0 ]; then
                echo "🛡️ Immune System Passed. Committing..." | tee -a $LOG_FILE
                
                # Generate Persona-based Commit Message
                if [ -f "scripts/generate_commit_msg.py" ]; then
                    COMMIT_MSG=$(python3 scripts/generate_commit_msg.py 2>>$LOG_FILE)
                else
                    COMMIT_MSG="Ralph: Routine update and evolution."
                fi
                
                git add .
                git commit -m "$COMMIT_MSG" >> $LOG_FILE 2>&1
                # git push origin main
            else
                echo "❌ Immune System Rejected Changes (Exit Code: $IMMUNE_EXIT). Reverting..." | tee -a $LOG_FILE
                echo "Check $LOG_FILE for details."
                git reset --hard HEAD >> $LOG_FILE 2>&1
            fi
        else
            echo "⚠️ Immune system script not found. Skipping validation (Risky)." | tee -a $LOG_FILE
        fi
        
    elif [ $EXIT_CODE -eq 42 ]; then
        # 4. Dependency Fix Path
        echo "🔧 Installing new dependencies..." | tee -a $LOG_FILE
        pip install -r requirements.txt
        
    else
        # 5. Failure Path
        echo "💀 Process Crashed. Reverting state to be safe..." | tee -a $LOG_FILE
        git reset --hard HEAD
    fi
    
    # 6. Sleep to prevent API rate limit spam
    echo "💤 Sleeping..."
    sleep 10
done
