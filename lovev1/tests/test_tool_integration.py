import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.deep_agent_engine import DeepAgentEngine
from core.tool_registry import ToolRegistry
from core.thought_chain import start_chain, get_current_chain, NodeStatus
from core.nodes.execution import tool_execution_node
from langchain_core.messages import ToolMessage
from display import create_active_tool_panel
from rich.console import Console

class TestToolIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = MagicMock(spec=ToolRegistry)
        # self.registry.__bool__.return_value = True # Removed invalid attr
        self.registry.__len__.return_value = 1
        self.registry.get_all_tool_schemas.return_value = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {"arg1": {"type": "string"}}}
            }
        ]
        self.engine = DeepAgentEngine(api_url="http://mock", tool_registry=self.registry)

    def test_adapt_tools_format(self):
        """Verify _adapt_tools_for_deepagent returns expected format."""
        formatted = self.engine._adapt_tools_for_deepagent()
        self.assertIn("Tool Name: `test_tool`", formatted)
        self.assertIn("Description: A test tool", formatted)
        self.assertIn("Arguments JSON Schema:", formatted)
        self.assertIn('"arg1":', formatted)

    def test_panel_rendering(self):
        """Verify panel creation does not crash."""
        tools = [
            {'name': 'test_tool', 'args': {'arg1': 'val'}, 'status': 'running'},
            {'name': 'done_tool', 'status': 'complete', 'result': 'ok'}
        ]
        panel = create_active_tool_panel(tools)
        # Just ensure it renders without error
        with open(os.devnull, "w", encoding="utf-8") as f:
             Console(file=f).print(panel)
        self.assertIsNotNone(panel)

class TestTracing(unittest.IsolatedAsyncioTestCase):
    async def test_execution_node_tracing(self):
        """Verify tool execution adds steps to thought chain."""
        start_chain("Test Chain")
        
        # Mock state and registry
        state = {"messages": [], "loop_count": 0}
        
        # Mock a tool call in the last message (using proper object structure if possible, or mocking)
        class MockMessage:
            tool_calls = [{"name": "mock_tool", "args": {}, "id": "call_1"}]
            content = ""
            
        state["messages"] = [MockMessage()]
        
        # Mock internal functions of execution.py
        with patch('core.nodes.execution._get_tool_from_registry') as mock_get_tool, \
             patch('core.nodes.execution._safe_execute_tool') as mock_exec:
            
            mock_get_tool.return_value = lambda: "success"
            mock_exec.return_value = "Tool Output"
            
            await tool_execution_node(state)
            
            chain = get_current_chain()
            self.assertIsNotNone(chain)
            # Check if a node with "Executing tool: mock_tool" exists
            found = False
            for node in chain.nodes.values():
                if "Executing tool: mock_tool" in node.content:
                    found = True
                    self.assertEqual(node.status, NodeStatus.SUCCESS)
                    break
            self.assertTrue(found, "Tracing step for tool execution not found")

if __name__ == '__main__':
    unittest.main()
