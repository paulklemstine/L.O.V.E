@echo off
wsl bash -c "cd ~/L.O.V.E/love2 && python3 tests/verify_tools.py > debug_output.txt 2>&1"
