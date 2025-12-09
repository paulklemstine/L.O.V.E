"""
Secure AI Fundamentals Research & Evolution Utility
Provides secure command execution and project management helpers
for the multi-phase AI fundamentals evolution plan.
"""

import subprocess
import os
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def execute_secure_command(command: str, *args: str) -> Dict[str, any]:
    """
    Execute external command securely using shlex.split to avoid shell=True.

    Args:
        command: Base command to execute
        *args: Additional arguments for the command

    Returns:
        Dict containing stdout, stderr, returncode

    Security:
        - Uses shlex.split instead of shell=True
        - Avoids command injection through proper argument separation
    """
    try:
        # Split command safely using shlex
        cmd_list = [command] + list(args)
        logger.info(f"Executing: {' '.join(cmd_list)}")

        # Execute process
        process = subprocess.Popen(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = process.communicate()

        return {"stdout": stdout, "stderr": stderr, "returncode": process.returncode}

    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}")
        return {"stdout": "", "stderr": str(e), "returncode": 1}


def generate_scope_matrix(
    algorithms: List[str], efforts: List[int], impacts: List[int]
) -> Dict[str, Dict[str, int]]:
    """
    Generate Scope Matrix prioritizing areas based on impact vs. effort.

    Args:
        algorithms: List of algorithm names to evaluate
        efforts: List of effort scores (1-5, where 1 is low effort)
        impacts: List of impact scores (1-5, where 5 is high impact)

    Returns:
        Prioritized matrix as dictionary
    """
    if len(algorithms) != len(efforts) or len(algorithms) != len(impacts):
        raise ValueError("Algorithm, effort and impact lists must have same length")

    matrix = {}

    for algo, effort, impact in zip(algorithms, efforts, impacts):
        # Calculate priority score (higher impact / lower effort)
        priority_score = impact / (effort if effort > 0 else 1)
        matrix[algo] = {
            "effort": effort,
            "impact": impact,
            "priority_score": priority_score,
        }

    # Sort algorithms by priority score descending
    sorted_matrix = dict(
        sorted(matrix.items(), key=lambda item: item[1]["priority_score"], reverse=True)
    )

    return sorted_matrix


def load_api_key(key_name: str, env_var: str = None) -> Optional[str]:
    """
    Securely load API key from environment variables.

    Args:
        key_name: Name to reference the key in code
        env_var: Optional environment variable name. Defaults to f"{KEY_NAME}_API_KEY"

    Returns:
        API key string or None if not found
    """
    if not env_var:
        env_var = f"{key_name.upper()}_API_KEY"

    if key_name in os.environ:
        return os.environ[key_name]

    if env_var in os.environ:
        return os.environ[env_var]

    logger.warning(f"API key '{key_name}' not found in environment variables")
    return None


# Example usage
if __name__ == "__main__":
    # Secure command execution example
    result = execute_secure_command("ls", "-l", "/tmp")
    logger.info(f"Command output: {result['stdout']}")

    # Scope matrix generation example
    algorithms = ["CNN", "Transformer", "GAN", "Decision Tree"]
    efforts = [3, 5, 4, 2]  # 1-5 scale (1=low effort)
    impacts = [5, 5, 4, 3]  # 1-5 scale (5=high impact)

    scope_matrix = generate_scope_matrix(algorithms, efforts, impacts)
    print("\nScope Matrix:")
    for algo, details in scope_matrix.items():
        print(
            f"{algo}: Effort={details['effort']}, Impact={details['impact']}, Priority={details['priority_score']:.2f}"
        )

    # Secure API key loading example
    api_key = load_api_key("ARXIV")
    if api_key:
        logger.info("ArXiv API key loaded successfully")
    else:
        logger.info(
            "ArXiv API key not configured - proceeding without remote literature fetch"
        )
