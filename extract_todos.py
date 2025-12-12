from code_analyzer import code_analyzer

def extract_todos(file_path):
    """
    Extracts and prints all TODOs from a given file.

    Args:
        file_path (str): The path to the file to analyze.
    """
    findings = code_analyzer(file_path)

    if not findings:
        print(f"No findings in {file_path}")
        return

    todos = [f for f in findings if f['keyword'].upper() == 'TODO']

    if not todos:
        print(f"No TODOs found in {file_path}")
        return

    print(f"Found {len(todos)} TODOs in {file_path}:\n")
    for todo in todos:
        print(f"  Line {todo['line_number']}: {todo['line']}")
    print("\n")

if __name__ == "__main__":
    extract_todos("love.py")
