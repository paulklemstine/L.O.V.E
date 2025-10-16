import asyncio
from core.ethereum.discovery import find_high_value_contracts
from core.ethereum.static_analysis import analyze_contract_vulnerabilities
from core.ethereum.attack_generator import generate_attack_vector
from core.ethereum.simulator import simulate_attack, SimulationError
from core.ethereum.utils import get_web3_instance, get_contract_source_code
from core.tools import ToolRegistry, SecureExecutor # Assuming these are the correct imports

class EthereumAgent:
    """
    An agent dedicated to Ethereum security analysis. It orchestrates the entire
    pipeline from discovery to simulation and notification.
    """
    def __init__(self, tool_registry: ToolRegistry, executor: SecureExecutor, node_url: str):
        self.tool_registry = tool_registry
        self.executor = executor
        self.node_url = node_url
        self.w3 = get_web3_instance(node_url)

    async def run_analysis_pipeline(self, scan_range: int = 200):
        """
        Executes the full, end-to-end analysis pipeline.
        """
        print("--- Starting Ethereum Analysis Pipeline ---")
        if not self.w3.is_connected():
            print(f"Error: Could not connect to Ethereum node at {self.node_url}")
            return

        # 1. Discover high-value contracts
        # In a real system, these tool calls would likely go through the SecureExecutor
        # for consistency, but calling them directly is fine for this agent's purpose.
        discovered_contracts = await find_high_value_contracts(self.w3, scan_range)

        if not discovered_contracts:
            print("Pipeline finished: No high-value contracts found in the specified block range.")
            return

        print(f"\n--- Analysis Phase: Found {len(discovered_contracts)} contracts to analyze. ---")
        for contract_addr in discovered_contracts:
            print(f"\n--- Analyzing Contract: {contract_addr} ---")

            # 2. Get source code
            source_code = get_contract_source_code(contract_addr)
            if not source_code:
                print(f"Skipping {contract_addr}: Could not fetch source code.")
                continue

            # 3. Run static analysis
            slither_report = await analyze_contract_vulnerabilities(contract_addr)
            if slither_report.get("error"):
                print(f"Skipping {contract_addr}: Slither analysis failed. {slither_report['error']}")
                continue

            # 4. Generate an attack vector with the LLM
            attack_plan = await generate_attack_vector(contract_addr, source_code, slither_report)
            if not attack_plan:
                print(f"No viable attack vector generated for {contract_addr}.")
                continue

            # 5. Simulate the attack
            try:
                simulation_result = await simulate_attack(attack_plan, contract_addr, self.node_url)

                # 6. If simulation is successful, notify the creator
                if simulation_result and simulation_result.get("success"):
                    print(f"!!! SUCCESSFUL ATTACK SIMULATED ON {contract_addr} !!!")

                    # Use the executor to call the registered notification tool
                    await self.executor.execute(
                        "notify_creator",
                        self.tool_registry,
                        contract_address=contract_addr,
                        attack_name=simulation_result.get("attack_name"),
                        simulation_log=simulation_result.get("log")
                    )
            except SimulationError as e:
                print(f"Simulation for {contract_addr} failed: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during simulation for {contract_addr}: {e}")

        print("\n--- Ethereum Analysis Pipeline Finished ---")

async def main():
    """A main function for testing the agent's pipeline."""
    # This setup is for demonstration. A real application would integrate this
    # into the main agent loop.

    # Mock tools and registry
    async def mock_notify_creator(**kwargs):
        print("\n--- MOCK NOTIFICATION ---")
        print("Creator has been notified of the successful simulation:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        print("-------------------------\n")

    registry = ToolRegistry()
    registry.register_tool("notify_creator", mock_notify_creator)

    executor = SecureExecutor()

    # Requires a node URL. You can use Infura, Alchemy, or a local node.
    # The agent will fail gracefully if it cannot connect.
    NODE_URL = os.environ.get("ETHEREUM_NODE_URL", "")

    if not NODE_URL:
        print("ETHEREUM_NODE_URL environment variable not set. Exiting test.")
        return

    agent = EthereumAgent(tool_registry=registry, executor=executor, node_url=NODE_URL)
    await agent.run_analysis_pipeline(scan_range=100) # smaller range for testing

if __name__ == "__main__":
    # Note: This test is comprehensive and requires:
    # 1. ETHEREUM_NODE_URL env var
    # 2. ETHERSCAN_API_KEY env var
    # 3. 'slither-analyzer' installed
    # 4. 'ganache-cli' installed
    # 5. 'core.llm_api' to be functional
    import os
    asyncio.run(main())