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
    research_prompt = f"""
    Based on the following project context, please generate a list of 3-5 high-level, cutting-edge AI and software engineering topics that would be most relevant for improving this project.
    Focus on areas like new agent architectures, memory systems, performance optimizations, or novel AI capabilities.
    Return a simple list of topics.

    {project_context}
    """

    research_topics_str = await run_llm(research_prompt)
    research_topics = [topic.strip() for topic in research_topics_str.strip().split('\n') if topic.strip()]

    # Step 2: Conduct web research on the brainstormed topics
    research_results = ""
    for topic in research_topics:
        search_summary, _ = await perform_webrequest(f"latest research and techniques in {topic} 2025")
        research_results += f"--- Research on: {topic} ---\n{search_summary}\n\n"

    # Step 3: Generate user stories based on context and research
    story_generation_prompt = f"""
    You are an expert AI software architect. Your task is to create a "book of user stories" to guide the evolution of an AI project named L.O.V.E.

    First, review the project's current state based on the provided file contents.
    Next, consider the latest AI research findings provided.

    Based on all this information, generate a list of 3-5 concrete, actionable user stories. Each story should represent a single, implementable feature or refactoring.
    The goal is to incrementally advance the project's capabilities, making it more intelligent, autonomous, and robust.

    Return the output as a JSON array of objects, where each object has a "title" and a "description". Do not include any other text or explanations.

    Example format:
    [
      {{"title": "Implement a Long-Term Memory Module", "description": "Create a new module `core/memory.py` that uses a vector database to store and retrieve information over long periods, improving the agent's contextual awareness."}},
      {{"title": "Refactor Tool Usage with a ReAct Engine", "description": "Replace the current ad-hoc tool execution logic with a formal Thought-Action-Observation loop in `core/gemini_react_engine.py` to improve reasoning and reliability."}}
    ]

    --- PROJECT CONTEXT ---
    {project_context}

    --- CUTTING-EDGE RESEARCH ---
    {research_results}
    """

    stories_json_str = await run_llm(story_generation_prompt, is_source_code=True)

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
