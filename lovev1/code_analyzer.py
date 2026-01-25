import re

def code_analyzer(file_path):
    """
    Analyzes a file to find lines containing specific keywords.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        list: A list of dictionaries, where each dictionary represents a finding
              and contains the line number, keyword, and the line content.
    """
    findings = []
    keywords = ["TODO", "description"]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                for keyword in keywords:
                    if re.search(r'\b' + keyword + r'\b', line, re.IGNORECASE):
                        findings.append({
                            "line_number": i,
                            "keyword": keyword,
                            "line": line.strip()
                        })
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return findings
