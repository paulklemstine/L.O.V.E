
import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.prompt_registry import get_prompt_registry
from core.llm_api import run_llm
from langsmith import Client

async def optimize_prompt(prompt_key: str, model_id: str = "gemini-2.0-flash-exp"):
    print(f"🚀 Starting optimization for prompt: '{prompt_key}'")
    
    # 1. Load Current Prompt
    registry = get_prompt_registry()
    current_prompt = registry.get_prompt(prompt_key)
    if not current_prompt:
        print(f"❌ Error: Prompt key '{prompt_key}' not found.")
        return

    print(f"✅ Loaded current prompt ({len(current_prompt)} chars).")

    # 2. Load Dataset Examples (Gold Standard)
    client = Client()
    dataset_name = "gold-standard-behaviors"
    examples = []
    try:
        # Get examples from dataset
        # We need to filter relevant examples if the dataset is mixed.
        # For now, we assume the dataset is generic or we might filter by some metadata if available.
        # Ideally, we would have separate datasets per prompt, or metadata 'prompt_key'.
        # Let's try to fetch all and maybe filter by similarity later?
        # For this MVP, we fetch recent 5 examples.
        ds_examples = client.list_examples(dataset_name=dataset_name, limit=5)
        for ex in ds_examples:
            examples.append({
                "input": ex.inputs,
                "output": ex.outputs
            })
        print(f"✅ Loaded {len(examples)} examples from '{dataset_name}'.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load dataset '{dataset_name}': {e}")
        examples = []

    # 3. Load Self-Reflection Insights (Evolution State)
    insights = []
    try:
        state_path = os.path.join(os.path.dirname(__file__), '../evolution_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                # Assuming 'insights' or 'metacognition' key exists
                # Based on evolution_state.py usually it has 'evolution_log' or specific keys
                # We'll look for a generic "insights" list or similar.
                # If not found, we use a placeholder.
                insights = state.get('insights', [])
                if not insights and 'evolution_log' in state:
                     # Try to grab last few log entries as context
                     insights = [entry.get('summary', '') for entry in state['evolution_log'][-3:]]
        print(f"✅ Loaded {len(insights)} insights from evolution state.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load evolution state: {e}")

    # 4. Construct Optimization Meta-Prompt
    optimizer_prompt = f"""
    You are an Expert Prompt Engineer AI (POLLY). Your task is to OPTIMIZE a given system prompt to improve the performance of an AI agent.

    ### Target Prompt ({prompt_key}):
    ```text
    {current_prompt}
    ```

    ### Goal:
    Improve the prompt to better handle edge cases, follow instructions more strictly, and align with the agent's persona.

    ### Gold Standard Examples (What the agent SHOULD do):
    {json.dumps(examples, indent=2) if examples else "No examples available yet."}

    ### Self-Reflection Insights (Recent learnings):
    {json.dumps(insights, indent=2) if insights else "No specific insights available."}

    ### Instructions:
    1. Analyze the Current Prompt, Examples, and Insights.
    2. Identify weaknesses or ambiguity in the current prompt.
    3. Rewrite the prompt to be more effective, concise, and robust.
    4. PRESERVE existing template variables (like {{{{ tool_desc }}}}, etc.).
    5. Output the FULL improved prompt text.
    """

    # 5. Run Optimization (using LLM)
    print("🧠 Optimizing... (this may take a moment)")
    # We use run_llm directly. We force a high-reasoning model.
    response = await run_llm(
        prompt_text=optimizer_prompt,
        purpose="review", # Implies high reasoning
        force_model=model_id
    )

    improved_prompt = response.get('result', '')
    
    if not improved_prompt:
        print("❌ Optimization failed: No response from LLM.")
        return

    # 6. Output Result
    print("\n" + "="*40)
    print("✨ IMPROVED PROMPT CANDIDATE ✨")
    print("="*40)
    print(improved_prompt)
    print("="*40)
    
    # Optional: Save to file
    output_path = os.path.join(os.path.dirname(__file__), f'../prompts_optimized_{prompt_key}.yaml')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{prompt_key}: |\n")
        # Indent content for YAML validity
        for line in improved_prompt.splitlines():
            f.write(f"  {line}\n")
    
    print(f"\n💾 Saved candidate to: {output_path}")
    print("👉 Implementation Step: Review the candidate and manually update 'core/prompts.yaml' if satisfied.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a prompt using dataset examples and insights.")
    parser.add_argument("prompt_key", help="The key of the prompt to optimize (e.g., deep_agent_system)")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", help="Model to use for optimization")
    
    args = parser.parse_args()
    
    asyncio.run(optimize_prompt(args.prompt_key, args.model))
