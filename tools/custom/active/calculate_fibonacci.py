"""
Auto-fabricated tool: calculate_fibonacci
Created: 2026-01-28T21:30:00
Capability: Calculate Fibonacci numbers
Status: ACTIVE - Promoted for testing
"""

from core.tool_registry import tool_schema


@tool_schema
def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    This tool computes Fibonacci numbers using an efficient iterative approach.
    Useful for mathematical computations and demonstrations.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Example:
        result = calculate_fibonacci(10)  # Returns 55
    """
    if n < 0:
        return 0
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
