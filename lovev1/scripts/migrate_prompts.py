import os
import sys
import yaml
from langchain import hub
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.prompt_manager import PromptManager

def migrate_prompts():
    print("Starting prompt migration to LangChain Hub...")
    prompts_path = os.path.join(os.path.dirname(__file__), '../core/prompts.yaml')
    
    # Load raw prompts
    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {prompts_path} not found.")
        return

    # Owner for the hub (repo handle)
    # Ideally this is 'love-agent' or the user's handle.
    # We will use 'love-agent' as a placeholder or read from env.
    repo_handle = os.environ.get("LANGCHAIN_HUB_REPO", "love-agent")
    
    print(f"Pushing prompts to handle: {repo_handle}...")

    for key, value in prompts_data.items():
        if not isinstance(value, str):
            print(f"Skipping non-string prompt key: {key}")
            continue
            
        print(f"Processing '{key}'...")
        
        # Determine if it's a chat prompt or string prompt (heuristic)
        # Most in prompts.yaml seem to be raw templates.
        # We'll use PromptTemplate by default.
        
        try:
            # Create a simple PromptTemplate
            # We assume the YAML content is the template string
            prompt = PromptTemplate.from_template(value)
            
            # Construct the repo path: handle/key
            repo_path = f"{repo_handle}/{key}"
            
            # Push to hub
            hub.push(repo_path, prompt)
            print(f"✅ Successfully pushed {repo_path}")
            
        except Exception as e:
            print(f"❌ Failed to push {key}: {e}")

    print("Migration complete.")

if __name__ == "__main__":
    migrate_prompts()
