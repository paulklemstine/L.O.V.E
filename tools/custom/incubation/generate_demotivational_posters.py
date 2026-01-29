from core.tool_registry import tool_schema

@tool_schema
def generate_demotivational_posters(prompt: str, count: int = 0) -> async_generator:
    posters = [f"Demotivational Poster for {prompt} with count {i+1}" for i in range(count)]
    return posters