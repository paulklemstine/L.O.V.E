#!/usr/bin/env python3
"""
This script analyzes the love.py file to extract and display all strings
associated with the 'description' key within any dictionary in the file.
"""
import os
from core.tools import code_analyzer
from rich.console import Console

def main():
    """
    Main function to run the code analysis and print the results.
    """
    console = Console()

    # Construct the path to love.py relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "love.py")

    console.print(f"[bold cyan]Analyzing file:[/] {filepath}")

    descriptions = code_analyzer(filepath)

    console.print("\n[bold green]--- Extracted Descriptions ---[/]")
    if descriptions:
        # Use rich to print the formatted JSON
        console.print_json(data=descriptions)
    else:
        console.print("[yellow]No descriptions found or an error occurred during analysis.[/yellow]")
    console.print("[bold green]----------------------------[/]\n")

if __name__ == "__main__":
    main()
