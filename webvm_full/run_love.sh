#!/bin/bash
# L.O.V.E. WebVM startup script
# This script ensures Python doesn't add the current directory to sys.path

# Change to L.O.V.E. directory
cd /root/L.O.V.E

# Run love.py with -P flag (don't prepend current dir to sys.path)
# -u for unbuffered output
# Pass all script arguments through
exec python3 -P -u love.py "$@"
