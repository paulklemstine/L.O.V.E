import json
from langchain_core.language_models.llms import LLM
from deepagents import create_deep_agent
import logging

class VLLMWrapper(LLM):
    """A custom LangChain wrapper for our async LocalVLLMClient."""
    vllm_client: "LocalVLLMClient"

    @property
    def _llm_type(self) -> str:
        return "custom_vllm_wrapper"

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        """Synchronous call to the async vLLM client."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(self.vllm_client.run(prompt), loop)
            result = future.result(timeout=600)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logging.error(f"Error in VLLMWrapper _call: {e}")
            return f"Error: {e}"

    async def _acall(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        """Asynchronous call to the vLLM client."""
        try:
            result = await self.vllm_client.run(prompt)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logging.error(f"Error in VLLMWrapper _acall: {e}")
            return f"Error: {e}"

class DeepAgentEngine:
    def __init__(self, vllm_client, tools):
        self.vllm_client = vllm_client
        self.tools = tools
        self.vllm_llm = VLLMWrapper(vllm_client=self.vllm_client)

    def run(self, system_prompt, user_prompt="Proceed with the next single strategic command."):
        agent = create_deep_agent(
            llm=self.vllm_llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        return result["messages"][-1].content
