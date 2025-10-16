import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from web3 import Web3

from core.agents.ethereum_agent import EthereumAgent
from core.tools import ToolRegistry, SecureExecutor

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_web3():
    """Fixture for a mock Web3 instance."""
    mock = MagicMock()
    mock.is_connected.return_value = True
    mock.eth.block_number = 10000
    # Mock balance to be over the threshold
    mock.eth.get_balance.return_value = Web3.to_wei(150, 'ether')
    # Mock get_code to indicate a contract exists
    mock.eth.get_code.return_value = b'0x...some_bytecode'
    return mock

@pytest.fixture
def tool_registry_and_executor():
    """Fixture for a ToolRegistry and SecureExecutor."""
    registry = ToolRegistry()
    executor = SecureExecutor()

    # Register a mock notification tool
    mock_notify = AsyncMock(return_value="Notification sent.")
    registry.register_tool("notify_creator", mock_notify)

    return registry, executor, mock_notify


@patch('core.agents.ethereum_agent.find_high_value_contracts')
@patch('core.agents.ethereum_agent.get_contract_source_code')
@patch('core.agents.ethereum_agent.analyze_contract_vulnerabilities')
@patch('core.agents.ethereum_agent.generate_attack_vector')
@patch('core.agents.ethereum_agent.simulate_attack')
async def test_ethereum_agent_full_pipeline_success(
    mock_simulate_attack,
    mock_generate_attack,
    mock_analyze_vulns,
    mock_get_source,
    mock_find_contracts,
    tool_registry_and_executor,
    mock_web3
):
    """
    Tests the full, successful execution path of the EthereumAgent's pipeline.
    """
    # --- Arrange ---
    # Mock the return values of our pipeline functions
    mock_find_contracts.return_value = {"0x1234567890123456789012345678901234567890"}
    mock_get_source.return_value = "contract {}"
    mock_analyze_vulns.return_value = {"success": True, "results": {"detectors": []}}
    mock_generate_attack.return_value = {"attack_name": "Test Attack", "attack_steps": ["Do something."]}
    mock_simulate_attack.return_value = {"success": True, "attack_name": "Test Attack", "log": ["Simulation log."]}

    registry, executor, mock_notify_tool = tool_registry_and_executor

    agent = EthereumAgent(
        tool_registry=registry,
        executor=executor,
        node_url="mock_url"
    )

    # Patch the agent's w3 instance directly to ensure it uses our mock
    agent.w3 = mock_web3

    # Mock the executor's execute method to intercept the final call
    executor.execute = AsyncMock()

    # --- Act ---
    await agent.run_analysis_pipeline(scan_range=10)

    # --- Assert ---
    mock_find_contracts.assert_called_once()
    mock_get_source.assert_called_once()
    mock_analyze_vulns.assert_called_once()
    mock_generate_attack.assert_called_once()
    mock_simulate_attack.assert_called_once()

    # Check that the notification tool was called via the executor
    executor.execute.assert_called_once_with(
        'notify_creator',
        registry,
        contract_address="0x1234567890123456789012345678901234567890",
        attack_name="Test Attack",
        simulation_log=["Simulation log."]
    )


@patch('core.agents.ethereum_agent.find_high_value_contracts')
async def test_ethereum_agent_no_contracts_found(
    mock_find_contracts,
    tool_registry_and_executor,
    mock_web3
):
    """
    Tests that the pipeline gracefully stops if no contracts are found.
    """
    # --- Arrange ---
    mock_find_contracts.return_value = set() # No contracts found

    registry, executor, mock_notify_tool = tool_registry_and_executor
    executor.execute = AsyncMock() # Mock the execute method
    agent = EthereumAgent(tool_registry=registry, executor=executor, node_url="mock_url")
    agent.w3 = mock_web3

    # --- Act ---
    await agent.run_analysis_pipeline()

    # --- Assert ---
    mock_find_contracts.assert_called_once()
    executor.execute.assert_not_called()