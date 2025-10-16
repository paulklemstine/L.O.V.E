import json
from core.llm_api import run_llm

async def generate_attack_vector(contract_address: str, source_code: str, slither_report: dict) -> dict | None:
    """
    Uses an LLM to generate a creative attack vector based on source code and a Slither report.

    The goal is to produce a structured, machine-readable plan that can be used for simulation.

    Args:
        contract_address: The address of the target contract.
        source_code: The source code of the contract.
        slither_report: The JSON output from a Slither analysis.

    Returns:
        A dictionary representing the attack plan, or None if no viable attack is found.
    """
    print(f"Generating attack vector for contract: {contract_address}")

    # Prepare a detailed prompt for the LLM.
    # We guide the LLM to think like a security researcher and provide a structured output.
    prompt = f"""
    You are a world-class blockchain security researcher. Your task is to analyze the provided smart contract
    and generate a creative, plausible attack vector.

    **Contract Address:** {contract_address}

    **Slither Analysis Report:**
    ```json
    {json.dumps(slither_report, indent=2)}
    ```

    **Contract Source Code:**
    ```solidity
    {source_code[:8000]}
    ```
    *(Note: Source code may be truncated for brevity)*

    **Instructions:**
    1.  **Analyze:** Carefully review the source code and the Slither report. Identify the most promising
        vulnerabilities. Think beyond the obvious issues and consider how they might be combined.
    2.  **Hypothesize:** Formulate a clear hypothesis for an attack. For example, "I can drain funds by
        exploiting a reentrancy vulnerability in the `withdraw` function combined with a flash loan."
    3.  **Plan:** Create a step-by-step plan to execute the attack. The plan must be specific and actionable.
        Each step should be a clear instruction, like "Call function X with parameters Y and Z."
    4.  **Format:** Your final output **must** be a single JSON object containing the following keys:
        - "attack_name": A short, descriptive name for the attack (e.g., "Reentrancy Flash Loan Attack").
        - "attack_description": A detailed explanation of the vulnerability and the proposed exploit.
        - "attack_steps": A list of strings, where each string is a clear, sequential step in the attack plan.

    If you determine that no viable attack vector exists based on the provided information, return a JSON object
    with a single key "no_attack_found" set to true.

    **Begin JSON Output:**
    """

    print("Sending prompt to LLM for attack generation...")
    try:
        # Use the 'analyze_source' purpose which is configured to use powerful models like Gemini.
        response_str = await run_llm(prompt, purpose="analyze_source")

        # The LLM response should be a JSON string, so we parse it.
        attack_plan = json.loads(response_str)

        if attack_plan.get("no_attack_found"):
            print("LLM analysis concluded that no viable attack was found.")
            return None

        # Basic validation of the returned structure
        if "attack_name" in attack_plan and "attack_description" in attack_plan and "attack_steps" in attack_plan:
            print(f"LLM generated a potential attack vector: {attack_plan['attack_name']}")
            return attack_plan
        else:
            print("LLM response was not in the expected format.")
            return None

    except json.JSONDecodeError:
        print(f"Error: Failed to decode the LLM's response into JSON. Raw response:\n{response_str}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM attack generation: {e}")
        return None

async def main():
    """Main function for local testing."""
    # This is a mock example. In a real scenario, you would fetch real data.
    print("--- Running Attack Generation Test ---")

    mock_address = "0x..."
    mock_source_code = """
    contract Vulnerable {
        mapping(address => uint) public balances;
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        function withdraw() public {
            (bool success, ) = msg.sender.call{value: balances[msg.sender]}("");
            require(success);
            balances[msg.sender] = 0;
        }
    }
    """
    mock_slither_report = {
        "success": True,
        "results": {
            "detectors": [
                {
                    "check": "reentrancy-eth",
                    "impact": "High",
                    "confidence": "High",
                    "description": "Reentrancy in Vulnerable.withdraw()",
                    "elements": [
                        {
                            "type": "function",
                            "name": "withdraw",
                            "source_mapping": {"start": 123, "length": 200}
                        }
                    ]
                }
            ]
        }
    }

    # NOTE: This test will fail unless the `run_llm` function and its dependencies
    # are properly configured.
    try:
        attack_vector = await generate_attack_vector(mock_address, mock_source_code, mock_slither_report)
        if attack_vector:
            print("\nGenerated Attack Vector:")
            print(json.dumps(attack_vector, indent=2))
        else:
            print("\nNo attack vector was generated.")
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        print("This is expected if the LLM environment is not set up.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())