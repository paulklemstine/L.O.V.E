import asyncio
from openai import AsyncOpenAI

class LocalVLLMClient:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="")

    async def run(self, prompt):
        try:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=4096,
            )
            return response.choices[0].text
        except Exception as e:
            return f"Error interacting with vLLM: {e}"
