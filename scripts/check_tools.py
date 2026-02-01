
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent dir to path
sys.path.insert(0, str(Path.cwd()))

load_dotenv()

try:
    from core.tool_adapter import get_adapted_tools
    tools = get_adapted_tools()
    
    print(f"Total tools: {len(tools)}")
    
    required = ["scout_influencers", "engage_with_influencer", "respond_to_comments"]
    missing = []
    
    for req in required:
        if req in tools:
            print(f"✅ {req} is available")
        else:
            print(f"❌ {req} is MISSING")
            missing.append(req)
            
    if missing:
        print("FAILED")
        sys.exit(1)
    else:
        print("SUCCESS")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
