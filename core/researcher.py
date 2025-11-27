import os
import json
from typing import List, Dict, Any

# Using local imports to avoid circular dependencies
# from core.llm_api import run_llm
# from network import perform_webrequest

def _get_project_context() -> str:
    """Gathers context from the current project's codebase."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    context_files = ['love.py', 'README.md', 'core/llm_api.py']

    context = "Project Context:\n\n"
    for file_path in context_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            context += f"--- Content of {file_path} ---\n"
            try:
                with open(full_path, 'r', errors='ignore') as f:
                    context += f.read() + "\n\n"
            except Exception:
                context += "Could not read file.\n\n"
    return context

async def generate_evolution_book() -> List[Dict[str, str]]:
    """
    Analyzes the codebase, researches cutting-edge AI, and generates a book of user stories.
    """
    # Import locally to avoid circular dependencies
    from core.llm_api import run_llm
    from network import perform_webrequest

    project_context = _get_project_context()

    # Step 1: Brainstorm research topics based on the codebase
    research_topics_dict = await run_llm(prompt_key="researcher_topics_brainstorm", prompt_vars={"project_context": project_context}, force_model=None)
    research_topics_str = research_topics_dict.get("result", "")
    research_topics = [topic.strip() for topic in research_topics_str.strip().split('\n') if topic.strip()]

    # Step 2: Conduct web research on the brainstormed topics
    research_results = ""
    for topic in research_topics:
        search_summary, _ = await perform_webrequest(f"latest research and techniques in {topic} 2025")
        research_results += f"--- Research on: {topic} ---\n{search_summary}\n\n"

    # Step 3: Generate user stories based on context and research
    stories_json_dict = await run_llm(prompt_key="researcher_story_generation", prompt_vars={"project_context": project_context, "research_results": research_results}, is_source_code=True, force_model=None)
    stories_json_str = stories_json_dict.get("result", "")

    try:
        # Clean up the response to ensure it's valid JSON
        if "```json" in stories_json_str:
            stories_json_str = stories_json_str.split("```json")[1].split("```")[0]

        user_stories = json.loads(stories_json_str)
        if isinstance(user_stories, list) and all(isinstance(story, dict) and 'title' in story and 'description' in story for story in user_stories):
            return user_stories
        else:
            return [] # Return empty list if format is incorrect
    except (json.JSONDecodeError, TypeError):
        return [] # Return empty list on failure
