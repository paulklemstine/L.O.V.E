import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from core.nodes.reasoning import reason_node, _apply_shadow_heuristics
from core.state import DeepAgentState
from langchain_core.messages import HumanMessage, AIMessage

class TestShadowMode(unittest.IsolatedAsyncioTestCase):

    def test_apply_shadow_heuristics(self):
        prompt = "System: You are an agent.\nUser: Give me a quick status update."
        messages = [HumanMessage(content="Give me a quick status update.")]
        modified = _apply_shadow_heuristics(prompt, messages)
        self.assertIn("[SHADOW MODE OVERRIDE]", modified)
        self.assertIn("concise", modified)

        prompt_no_trigger = "System: You are an agent.\nUser: Tell me a story."
        messages_no_trigger = [HumanMessage(content="Tell me a story.")]
        modified_no_trigger = _apply_shadow_heuristics(prompt_no_trigger, messages_no_trigger)
        self.assertEqual(prompt_no_trigger, modified_no_trigger)

    @patch('core.nodes.reasoning.run_llm', new_callable=AsyncMock)
    @patch('core.nodes.reasoning.stream_llm')
    async def test_reason_node_shadow_mode(self, mock_stream_llm, mock_run_llm):
        # Setup mocks
        mock_run_llm.return_value = {"result": "Shadow Response: All systems normal."}

        async def async_generator():
            yield "Normal Response: Greetings Creator..."
        mock_stream_llm.return_value = async_generator()

        # Setup State
        messages = [HumanMessage(content="Give me a quick status update.")]
        state: DeepAgentState = {
            "messages": messages,
            "shadow_mode": True,
            "shadow_log": [],
            "tool_schemas": [],
            "loop_count": 0
        }

        # Run Node
        result = await reason_node(state)

        # Verify Shadow Execution
        self.assertTrue(mock_run_llm.called)
        args, _ = mock_run_llm.call_args
        self.assertIn("[SHADOW MODE OVERRIDE]", args[0])

        # Verify State Update
        self.assertIn("shadow_log", result)
        self.assertEqual(len(result["shadow_log"]), 1)
        self.assertEqual(result["shadow_log"][0]["result"], "Shadow Response: All systems normal.")

        # Verify Normal Execution
        self.assertTrue(mock_stream_llm.called)
        self.assertEqual(result["messages"][0].content, "Normal Response: Greetings Creator...")

if __name__ == '__main__':
    unittest.main()
