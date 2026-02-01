import asyncio
import unittest
from unittest.mock import MagicMock, patch
from core.agents.creative_writer_agent import CreativeWriterAgent
from core.introspection.tool_gap_detector import ToolGapDetector
from core.agents.creator_command_agent import CreatorCommandAgent
from core.agents.evolutionary_agent import EvolutionaryAgent

class TestTemperatures(unittest.IsolatedAsyncioTestCase):
    async def test_creative_writer_agent_temperatures(self):
        agent = CreativeWriterAgent()
        mock_client = MagicMock()
        mock_client.generate_async = MagicMock(return_value=asyncio.Future())
        mock_client.generate_async.return_value.set_result("test result")
        
        with patch('core.agents.creative_writer_agent.get_llm_client', return_value=mock_client):
            # Test story generation (T=1.0)
            try:
                await agent._generate_story_content("voice", "theme", "mood", 280)
            except: pass
            args, kwargs = mock_client.generate_async.call_args
            self.assertEqual(kwargs.get('temperature'), 1.0)
            
            # Test reply generation (T=0.9)
            mock_client.generate_async.reset_mock()
            try:
                await agent._generate_reply_text("voice", "text", "author", "mood", 280)
            except: pass
            args, kwargs = mock_client.generate_async.call_args
            self.assertEqual(kwargs.get('temperature'), 0.9)

    async def test_reasoning_agent_temperatures(self):
        # 1. ToolGapDetector (T=0.2)
        mock_llm = MagicMock()
        mock_llm.generate_async = MagicMock(return_value=asyncio.Future())
        mock_llm.generate_async.return_value.set_result('{"functional_name": "test"}')
        
        detector = ToolGapDetector(llm_client=mock_llm)
        with patch('core.introspection.tool_gap_detector.add_tool_specification'):
            await detector.analyze_gap_and_specify("context")
            args, kwargs = mock_llm.generate_async.call_args
            self.assertEqual(kwargs.get('temperature'), 0.2)
            
        # 2. CreatorCommandAgent (T=0.2)
        mock_llm_json = MagicMock()
        mock_llm_json.generate_json_async = MagicMock(return_value=asyncio.Future())
        mock_llm_json.generate_json_async.return_value.set_result({"thought": "done", "action": "reply_to_creator", "action_input": {"message": "ok"}})
        
        with patch('core.agents.creator_command_agent.get_llm_client', return_value=mock_llm_json):
            agent = CreatorCommandAgent()
            await agent.process_command("test command")
            args, kwargs = mock_llm_json.generate_json_async.call_args
            self.assertEqual(kwargs.get('temperature'), 0.2)

if __name__ == '__main__':
    unittest.main()
