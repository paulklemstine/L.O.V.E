
import os
import re
import time
from datetime import datetime

def test_todo_logic():
    print("Testing Creator Command Logic...")
    
    # Mock inputs
    p_text = "Task: Fix the subspace transceiver."
    p_author_handle = "evildrgemini.bsky.social"
    is_creator = True
    
    # Logic extracted from tools_legacy.py
    keywords = ["task", "command", "order", "todo", "req", "feature"]
    if is_creator and any(k in p_text.lower() for k in keywords):
        try:
            # Extract content
            cmd_content = p_text
            for k in keywords:
                if f"{k}:" in p_text.lower():
                    parts = re.split(f"{k}:", p_text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        cmd_content = parts[1].strip()
                        break
            
            todo_file = "TODO_TEST_ARTIFACT.md" # Valid test file
            existing = ""
            if os.path.exists(todo_file):
                with open(todo_file, 'r', encoding='utf-8') as f:
                    existing = f.read()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_entry = f"- [ ] **CREATOR DECREE ({timestamp}):** {cmd_content} <!-- id: creator_{int(time.time())} -->\n"
            
            with open(todo_file, 'w', encoding='utf-8') as f:
                f.write(new_entry + existing)
            
            print(f"Success! Written to {todo_file}")
            print("Content:")
            with open(todo_file, 'r') as f:
                print(f.read())
                
            # Clean up
            os.remove(todo_file)
            
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("Logic condition failed.")

if __name__ == "__main__":
    test_todo_logic()
