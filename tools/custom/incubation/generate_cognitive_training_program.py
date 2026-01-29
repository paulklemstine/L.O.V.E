from core.tool_registry import tool_schema

@tool_schema
@asyncio.coroutine
def generate_cognitive_training_program(prompt: str, count: int = 0) -> dict:
    try:
        training_program = f"Training program for {prompt} with {count} sessions."
        return training_program
    except Exception as e:
        return {"error": str(e)}