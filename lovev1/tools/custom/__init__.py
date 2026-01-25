# tools/custom/__init__.py
"""
Story 4.1: Just-in-Time Tool Fabrication

This directory contains dynamically fabricated tools created by the agent
when it encounters problems it wasn't originally programmed to solve.

Tools in this directory are automatically loaded by ToolRegistry.refresh().
Each tool must be a Python file containing functions decorated with @tool_schema.

Example:
    tools/custom/pdf_parser.py
    tools/custom/specialized_scraper.py
"""

__all__ = []
