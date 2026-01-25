
import os

filepath = r"\\wsl.localhost\Ubuntu\home\raver1975\L.O.V.E\love.log"

try:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    error_lines = [line for line in lines[-2000:] if "ERROR" in line or "CRITICAL" in line or "Exception" in line or "Traceback" in line]
    
    if not error_lines:
        print("No errors found in the last 2000 lines.")
    else:
        print("Found errors:\n")
        print("".join(error_lines[-50:])) # Print last 50 errors

except Exception as e:
    print(f"Error reading file: {e}")

