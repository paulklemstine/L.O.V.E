
import pytest
import asyncio
import os
import json
from unittest.mock import MagicMock, patch
from core.pi_rpc_bridge import PiRPCBridge

@pytest.mark.asyncio
async def test_pi_bridge_spawn():
    """Test that PiRPCBridge constructs the command correctly."""
    
    # Mock subprocess to avoid actually running node if not built
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["", ""] # simulate no output and exit
        mock_process.stderr.readline.side_effect = [""]
        mock_popen.return_value = mock_process
        
        bridge = PiRPCBridge("/fake/dir")
        await bridge.start()
        
        assert bridge.running
        assert mock_popen.called
        args = mock_popen.call_args[0][0]
        assert args[0] == "node"
        assert "--mode" in args
        assert "rpc" in args
        assert "--provider" in args
        assert "vllm" in args
        
        await bridge.stop()
        assert not bridge.running

@pytest.mark.asyncio
async def test_pi_bridge_protocol():
    """Test sending and receiving JSON commands."""
    
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        
        # Simulate agent outputting a JSON event then EOF
        mock_process.stdout.readline.side_effect = [
            json.dumps({"type": "response", "command": "prompt", "success": True}) + "\n",
            ""
        ]
        mock_process.stderr.readline.return_value = ""
        mock_popen.return_value = mock_process
        
        bridge = PiRPCBridge("/fake/dir")
        
        # Capture events
        events = []
        async def on_event(data):
            events.append(data)
            
        bridge.set_callback(on_event)
        await bridge.start()
        
        # Give it a moment to process the mock stdout
        await asyncio.sleep(0.1)
        
        assert len(events) == 1
        assert events[0]["type"] == "response"
        assert events[0]["success"] is True
        
        await bridge.stop()
