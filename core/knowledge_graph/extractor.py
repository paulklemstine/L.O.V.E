import json
import ast
from typing import Callable
from core.knowledge_graph.prompts import KNOWLEDGE_EXTRACTION_PROMPT

class KnowledgeExtractor:
    """
    A class to extract knowledge from command outputs using an LLM.
    """
    def __init__(self, llm_api: Callable):
        self.llm_api = llm_api

    def extract_from_output(self, command_name: str, command_output: str) -> list:
        """
        Extracts entities and relationships from a command output.
        """
        if not command_output or not isinstance(command_output, str):
            return []

        prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(
            command_name=command_name,
            command_output=command_output
        )

        try:
            # Use the provided LLM function to get the structured data
            response = self.llm_api(prompt, purpose="knowledge_extraction")

            # The response is expected to be a string representation of a list of tuples
            # We need to parse this string to get the actual list
            extracted_triples = self._parse_llm_response(response)

            return extracted_triples

        except Exception as e:
            print(f"Error extracting knowledge from command output: {e}")
            raise e

    def _parse_llm_response(self, response: str) -> list:
        """
        Parses the LLM's string response to extract a list of triples.
        """
        try:
            # The response might be a JSON string, or a Python literal.
            # We'll try to handle both.
            response = response.strip()
            if response.startswith("[") and response.endswith("]"):
                # It looks like a list, so we can use ast.literal_eval
                return ast.literal_eval(response)
            else:
                return []

        except (ValueError, SyntaxError) as e:
            print(f"Error parsing LLM response: {e}")
            return []