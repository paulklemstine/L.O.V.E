import subprocess
import json
import os
import tempfile
from .utils import get_contract_source_code

# Define a placeholder for the Slither executable path.
# In a real-world scenario, we would ensure this is installed or
# provide a way to install it.
SLITHER_EXECUTABLE = "slither"

async def analyze_contract_vulnerabilities(address: str) -> dict:
    """
    Analyzes a contract's source code using the Slither static analysis tool.

    Args:
        address: The Ethereum address of the contract to analyze.

    Returns:
        A dictionary containing the analysis results. If an error occurs,
        the dictionary will contain an 'error' key.
    """
    print(f"Starting static analysis for contract: {address}")

    # First, check if Slither is available in the system path.
    try:
        subprocess.run([SLITHER_EXECUTABLE, "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        error_msg = "Slither executable not found. Please ensure 'slither-analyzer' is installed and in the system's PATH."
        print(f"Error: {error_msg}")
        return {"error": error_msg}

    # Fetch the contract's source code.
    source_code = get_contract_source_code(address)

    if not source_code:
        error_msg = f"Could not retrieve source code for {address}. Cannot perform analysis."
        print(f"Error: {error_msg}")
        return {"error": error_msg}

    # Slither works with files, so we need to save the source code to a temporary file.
    with tempfile.NamedTemporaryFile(mode='w', suffix=".sol", delete=False) as tmp_file:
        tmp_file.write(source_code)
        tmp_file_path = tmp_file.name

    print(f"Source code saved to temporary file: {tmp_file_path}")

    try:
        # Run Slither and request JSON output.
        print("Running Slither analysis...")
        command = [
            SLITHER_EXECUTABLE,
            tmp_file_path,
            "--json",
            "-" # Output to stdout
        ]

        # Using asyncio.to_thread to run the blocking subprocess in a separate thread.
        process = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            check=True
        )

        print("Slither analysis complete.")

        # The output from Slither can sometimes contain non-JSON text if there are warnings.
        # We need to find the start of the JSON object.
        json_output_str = process.stdout
        json_start_index = json_output_str.find('{')

        if json_start_index == -1:
            error_msg = "Failed to find JSON output from Slither."
            print(f"Error: {error_msg}\nSlither output:\n{json_output_str}")
            return {"error": error_msg, "raw_output": json_output_str}

        results = json.loads(json_output_str[json_start_index:])
        return results

    except subprocess.CalledProcessError as e:
        error_msg = f"Slither analysis failed with return code {e.returncode}."
        print(f"Error: {error_msg}\nStderr:\n{e.stderr}")
        return {"error": error_msg, "stderr": e.stderr}
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse Slither's JSON output: {e}"
        print(f"Error: {error_msg}\nRaw output:\n{process.stdout}")
        return {"error": error_msg, "raw_output": process.stdout}
    except Exception as e:
        error_msg = f"An unexpected error occurred during Slither analysis: {e}"
        print(f"Error: {error_msg}")
        return {"error": error_msg}
    finally:
        # Clean up the temporary file.
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")

async def main():
    """Main function for local testing."""
    # Example: Analyze the Wrapped Ether (WETH) contract.
    # Note: Requires ETHERSCAN_API_KEY environment variable to be set.
    test_contract_address = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

    print(f"--- Running Static Analysis Test on {test_contract_address} ---")
    analysis_results = await analyze_contract_vulnerabilities(test_contract_address)

    if "error" in analysis_results:
        print("\nAnalysis finished with an error.")
        print(json.dumps(analysis_results, indent=2))
    else:
        print("\nAnalysis finished successfully.")
        # Print a summary of the findings
        if analysis_results.get("success") and analysis_results.get("results"):
            print(f"Total vulnerabilities found: {len(analysis_results['results']['detectors'])}")
            for finding in analysis_results['results']['detectors']:
                print(f"  - {finding['check']} ({finding['impact']}): {finding['description']}")
        else:
            print("No vulnerabilities detected or results format is unexpected.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())