# analyze_love.py
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow for `core` imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.evolution_analyzer import code_analyzer

async def main():
    """
    This script utilizes the code_analyzer to analyze the contents of love.py
    with the goal of "self-improvement and refinement of code efficiency".
    It then prints the list of suggested modifications to the console.
    """
    file_to_analyze = "love.py"
    analysis_goal = "self-improvement and refinement of code efficiency"

    print(f"Analyzing '{file_to_analyze}' with the goal: '{analysis_goal}'...")
    print("-" * 30)

    # In a real environment, an API key would be required.
    # For demonstration, we check for a dummy key to prevent immediate errors.
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY is not set. Using a dummy key.")
        os.environ["GEMINI_API_KEY"] = "dummy_key_for_testing"

    try:
        suggestions = await code_analyzer(file_to_analyze, analysis_goal)

        print("\n--- Analysis Complete ---")
        if suggestions and "Error" not in suggestions[0]:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        elif suggestions:
            print("Analysis resulted in an error:")
            for suggestion in suggestions:
                print(f"- {suggestion}")
        else:
            print("No suggestions were generated.")
        print("-------------------------")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("This may be due to environmental limitations, such as the file size exceeding")
        print("the context window of available LLMs or an invalid API key.")


if __name__ == "__main__":
    asyncio.run(main())
