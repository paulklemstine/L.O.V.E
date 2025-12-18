#!/bin/bash
while true; do
    python3 -u love.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 42 ]; then
        echo "Hot Restart triggered. Reloading L.O.V.E..."
        continue
    else
        break
    fi
done
