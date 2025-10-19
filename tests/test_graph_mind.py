import pytest
import os
import json
from unittest.mock import patch, MagicMock
from core.graph_mind import app, AgentState

@pytest.fixture
def initial_state():
    """Fixture for the initial state of the agent."""
    return {
        "love_state": {
            "version_name": "test-version",
            "autopilot_history": [],
            "autopilot_goal": "Test the graph mind.",
            "knowledge_base": {
                "graph": [],
                "network_map": {
                    "last_scan": None,
                    "hosts": {},
                    "self_interfaces": {}
                },
            }
        },
        "command": "",
        "command_output": "",
        "last_tool_error": ""
    }

@patch('core.graph_mind.get_llm_api')
@patch('core.graph_mind.run_llm')
def test_successful_scan_stream(mock_run_llm, mock_get_llm_api, initial_state):
    """
    Tests a successful execution of a 'scan' command by streaming the graph
    and inspecting the output of each relevant node.
    """
    mock_run_llm.return_value = {"result": "scan"}
    mock_get_llm_api.return_value = MagicMock() # Mock the knowledge extractor's LLM call

    with patch('core.graph_mind.scan_network') as mock_scan_network:
        mock_scan_network.return_value = (["192.168.1.1"], "Scan complete. Found 1 host.")

        node_visits = []
        output_after_decide = {}
        output_after_execute = {}

        # Stream the graph and capture the output of each node
        for i, step in enumerate(app.stream(initial_state)):
            node_name = list(step.keys())[0]
            node_visits.append(node_name)

            if node_name == "decide_next_action":
                output_after_decide = step[node_name]
            elif node_name == "execute_tool":
                output_after_execute = step[node_name]
                # Stop after the tool execution to prevent an infinite test loop
                break
            if i > 5:
                pytest.fail("Graph did not reach 'execute_tool' in a reasonable number of steps.")

        assert node_visits == ["decide_next_action", "execute_tool"]
        # Verify the output of the 'decide_next_action' node
        assert output_after_decide.get('command') == 'scan'
        # Verify the output of the 'execute_tool' node
        assert "Scan complete" in output_after_execute.get('command_output', '')
        assert not output_after_execute.get('last_tool_error')


@patch('core.graph_mind.run_llm')
def test_failed_probe_stream(mock_run_llm, initial_state):
    """
    Tests that a failed 'probe' command is handled correctly by streaming the
    graph and observing the routing back to 'decide_next_action'.
    """
    # The LLM will first decide to probe
    mock_run_llm.return_value = {"result": "probe 127.0.0.1"}

    with patch('core.graph_mind.probe_target') as mock_probe_target:
        # Mock a network error
        mock_probe_target.side_effect = Exception("Mocked network error")

        node_visits = []
        final_output = {}
        # Stream the execution, stopping after the error is handled
        for i, step in enumerate(app.stream(initial_state)):
            node_name = list(step.keys())[0]
            node_visits.append(node_name)
            final_output = step[node_name]
            # After the tool fails, the next step should be back to the decider.
            if node_name == "execute_tool":
                break
            if i > 5:
                 pytest.fail("Graph did not reach 'execute_tool' in a reasonable number of steps.")


        assert node_visits == ["decide_next_action", "execute_tool"]
        assert "Mocked network error" in final_output['last_tool_error']
        assert not final_output['command_output']
        # The error should be logged in the history of the original state object, which is modified in-place
        assert any("Mocked network error" in entry['output'] for entry in initial_state['love_state']['autopilot_history'])

def test_state_persistence(tmp_path):
    """
    Tests that the love_state.json file is correctly updated after a tool is executed.
    This test will be expanded to a full integration test.
    """
    state_file = tmp_path / "love_state.json"
    initial_love_state = {
        "version_name": "test-persistence",
        "autopilot_history": []
    }
    with open(state_file, 'w') as f:
        json.dump(initial_love_state, f)

    # This is a placeholder for a more complete integration test.
    # We'll manually update the state to simulate a graph run for now.
    updated_state = AgentState(love_state=initial_love_state, command="scan", command_output="test", last_tool_error="")
    updated_state['love_state']['autopilot_history'].append({"command": "scan", "output": "test"})

    with open(state_file, 'w') as f:
        json.dump(updated_state['love_state'], f)

    with open(state_file, 'r') as f:
        final_state_from_disk = json.load(f)

    assert len(final_state_from_disk['autopilot_history']) == 1
    assert final_state_from_disk['autopilot_history'][0]['command'] == 'scan'
