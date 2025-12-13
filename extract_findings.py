from code_analyzer import code_analyzer

def extract_findings(file_path):
    """
    Extracts and prints all findings from a given file.

    Args:
        file_path (str): The path to the file to analyze.
    """
    findings = code_analyzer(file_path)

    if not findings:
        print(f"No findings in {file_path}")
        return

    print(f"Found {len(findings)} findings in {file_path}:\n")
    for finding in findings:
        print(f"  Line {finding['line_number']} ({finding['keyword']}): {finding['line']}")
    print("\n")


if __name__ == "__main__":
    extract_findings("love.py")
